// ns_cyl_sycl_demo.cpp
// Taylor-Couette cylinder NS — SYCL compute + Metal visualization.
// Inner cylinder (r0) rotates at U0; outer (R) is stationary.
// Particles rendered in the XY plane (top-down view) showing the flow pattern.

// ── metal-cpp (declarations only; implementations in *_metal_impl.cpp) ────────
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#ifdef nil
#  undef nil
#endif

// ── SDL2 ──────────────────────────────────────────────────────────────────────
#include <SDL2/SDL.h>
#include <SDL2/SDL_metal.h>

// ── SYCL + simulation ─────────────────────────────────────────────────────────
#include "ns_cyl_sycl.h"

// ── Standard ──────────────────────────────────────────────────────────────────
#include <charconv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <chrono>
#include <string_view>

// ═════════════════════════════════════════════════════════════════════════════
// Metal shaders
// ═════════════════════════════════════════════════════════════════════════════
static constexpr float kTrailAlpha = 0.015f;

static const char kMSL[] = R"msl(
#include <metal_stdlib>
using namespace metal;

// ── Trail fade ────────────────────────────────────────────────────────────────
vertex float4 fade_vert(uint vid [[vertex_id]])
{
    float2 pos[4] = {float2(-1,1), float2(1,1), float2(-1,-1), float2(1,-1)};
    return float4(pos[vid], 0, 1);
}
fragment float4 fade_frag(constant float& alpha [[buffer(0)]])
{
    return float4(0.f, 0.f, 0.f, alpha);
}

// ── Particles ─────────────────────────────────────────────────────────────────
struct VOut {
    float4 pos   [[position]];
    float  hue;
    float  depth;   // 0 = far wall of the cylinder, 1 = wall nearest the viewer
    float  psize [[point_size]];
};

// Must match kZoom in advect_particles(): render_buf coordinates are already
// scaled by it, so the rotated depth lands in [-kZoom, kZoom].
constant float kZoom     = 0.8f;
// Depth cue: the far half fades out and shrinks so it stops competing with the
// flow in front.  Raise kDepthFade toward 1 for a flatter, denser picture.
constant float kDepthFade = 0.10f;   // alpha of the farthest particles
constant float kSizeFar   = 1.4f;
constant float kSizeNear  = 3.2f;

static float3 hsv2rgb(float h)
{
    float3 rgb = clamp(abs(fmod(h*6.f + float3(0.f,4.f,2.f), 6.f) - 3.f) - 1.f, 0.f, 1.f);
    return mix(float3(1.f), rgb, 0.85f);
}

// Diverging palette for a signed quantity, 0.5 = at rest.  The neutral keeps
// enough luminance to stay visible against the black background, so still
// fluid reads as grey rather than disappearing; only the sign carries colour.
static float3 diverging(float t)
{
    float  s       = clamp(t*2.f - 1.f, -1.f, 1.f);
    float3 neutral = float3(0.50f, 0.52f, 0.58f);
    float3 down    = float3(0.15f, 0.50f, 1.00f);   // blue
    float3 up      = float3(1.00f, 0.38f, 0.14f);   // orange
    return mix(neutral, s < 0.f ? down : up, abs(s));
}

// pts = float4[]{x/R, z_norm, y/R, hue}  (zoom already applied)
// 3D orthographic: Ry(h) then Rx(v), cylinder axis vertical
// rot = {cos_h, sin_h, cos_v, sin_v}
vertex VOut ns_vert(uint                 vid [[vertex_id]],
                    const device float4* pts [[buffer(0)]],
                    constant float4&     rot [[buffer(1)]])
{
    float3 p  = pts[vid].xyz;
    float ch  = rot.x, sh = rot.y;
    float cv  = rot.z, sv = rot.w;
    float rx  =  ch*p.x + sh*p.z;
    float ry  =  sv*sh*p.x + cv*p.y - sv*ch*p.z;
    // Third component of the very same rotation -- the one the orthographic
    // projection throws away.  It is the distance along the view axis, so it is
    // exactly the depth cue we need.  Negate it if front and back read swapped.
    float rz  =  sv*p.y + cv*(ch*p.z - sh*p.x);

    VOut o;
    o.pos   = float4(rx, ry, 0.f, 1.f);
    o.hue   = pts[vid].w;
    o.depth = clamp(0.5f + 0.5f*rz/kZoom, 0.f, 1.f);
    o.psize = mix(kSizeFar, kSizeNear, o.depth);
    return o;
}

fragment float4 ns_frag(VOut in [[stage_in]],
                        constant int& mode [[buffer(0)]])
{
    // Squared so the falloff is concentrated on the far half: the front stays
    // at full strength while the back recedes into the trails behind it.
    float alpha = mix(kDepthFade, 1.f, in.depth*in.depth);
    // mode 0 is a Lagrangian marker -- an unordered label, so a cyclic hue.
    // Modes 1 and 2 are signed velocities and need a diverging palette.
    float3 rgb  = (mode == 0) ? hsv2rgb(in.hue) : diverging(in.hue);
    return float4(rgb, alpha);
}
)msl";

// ═════════════════════════════════════════════════════════════════════════════
// Demo
// ═════════════════════════════════════════════════════════════════════════════
static constexpr int kNR=32, kNZ=64, kNPHI=64;
static constexpr int kNP=32768;

static constexpr float kR0   = 1.0f;   // inner cylinder radius
static constexpr float kR    = 2.0f;   // outer cylinder radius

// Axial period.  A Taylor vortex is nearly square in cross-section, so its
// height is about the gap width d = kR - kR0 = 1.  z is periodic, so only
// whole wavelengths fit and one wavelength holds a counter-rotating pair:
// the vortex count is kLZ/d rounded to an even number.  Formally it is
// 2*round(k_c*kLZ/2pi) with the critical Taylor wavenumber k_c*d ~ 3.16.
// The flow is never seeded -- it starts from rest and the fastest growing
// mode wins over round-off noise -- so the count below is what you get.
static constexpr float kLZ = 2.0f;            //  2 vortices (k=3.14, best fit)
//static constexpr float kLZ = float(M_PI);     //  4 vortices (borderline: the
                                                //  box admits only k=2 or k=4,
                                                //  both far from k_c; 2 vortices
                                                //  are possible here as well)
//static constexpr float kLZ   = float(2*M_PI);   //  6 vortices (k=3.00)
//static constexpr float kLZ = 8.0f;            //  8 vortices (k=3.14, best fit)
//static constexpr float kLZ = float(3*M_PI);   // 10 vortices (k=3.33)
//static constexpr float kLZ = float(4*M_PI);   // 12 vortices (k=3.00), but
                                                // kNZ=64 leaves only ~5 cells
                                                // per vortex -- raise kNZ to 96
                                                // or 128 for a clean picture.

struct Demo {
    sycl::queue syclQ{
        []() {
            for (auto& plat : sycl::platform::get_platforms())
                for (auto& dev : plat.get_devices())
                    if (dev.is_gpu()) return dev;
            return sycl::device{sycl::cpu_selector_v};
        }(),
        sycl::property::queue::in_order{}};

    fdm::NSCylSycl<float> sim;

    float *part_px=nullptr, *part_py=nullptr, *part_pz=nullptr;
    float *color_buf=nullptr;
    float *render_buf=nullptr;   // float4 per particle: {x/R, y/R, z_norm, hue}

    MTL::Device*              dev     = nullptr;
    MTL::CommandQueue*        renderQ = nullptr;
    MTL::RenderPipelineState* pso     = nullptr;
    MTL::RenderPipelineState* fadePSO = nullptr;
    MTL::Buffer*              renderMetalBuf  = nullptr; // GPU-side view of render_buf
    NS::UInteger              renderBufOffset = 0;       // render_buf inside it
    bool                      zeroCopy        = false;   // no per-frame memcpy
    MTL::CommandBuffer*       prevCB         = nullptr;  // kept only to drain on exit
    CA::MetalLayer*           layer          = nullptr;

    // GPU-side handshake with SYCL (Metal backend only).  The render pass signals
    // renderDone, and the next frame's SYCL work is made to wait for it through
    // sycl::make_event, while the render pass waits for the SYCL upload through
    // sycl::get_native -- neither direction goes through the CPU.
    MTL::SharedEvent*         renderDone      = nullptr;
    uint64_t                  renderDoneValue = 0;
    bool                      interop         = false;

    uint32_t frame      = 0;
    bool     paused     = false;

    // Command line: --no-vsync frees the frame rate from the display refresh
    // (useful for measuring), --fps reports what it turns into.
    bool     vsync      = true;
    bool     showFps    = false;
    int      stepsPerFrame = 3;
    double   t_wait_drawable = 0, t_sycl = 0, t_render = 0;
    int      drawableW = 0, drawableH = 0;
    float    angle_h    = 0.2f;   // slight horizontal rotation to show 3D depth
    float    angle_v    = 0.0f;   // no vertical tilt — keep cylinder axis strict vertical

    // Trails are the drawable textures' own contents, kept by load action Load.
    // CAMetalLayer cycles through maximumDrawableCount of them, so each holds
    // every Nth frame of history -- clearing a single frame would wipe one
    // buffer and let the other N-1 bring their stale trails right back.  A
    // whole cycle has to be cleared, hence a countdown rather than a flag.
    int      clearFrames  = 1;    // set properly in init(), once layer is known
    int      clearCycle   = 3;

    // Rotating invalidates every trail on screen: they were drawn under the old
    // orientation and would smear across the new one.  Route all view changes
    // through here so none can forget to ask for the wipe.
    void rotate(float delta_h, float delta_v)
    {
        angle_h += delta_h;
        angle_v += delta_v;
        clearFrames = clearCycle;
    }

    int colorMode = fdm::NSCylSycl<float>::color_axial;

    static const char* color_name(int mode)
    {
        switch (mode) {
        case fdm::NSCylSycl<float>::color_axial:  return "axial velocity v_z";
        case fdm::NSCylSycl<float>::color_radial: return "radial velocity v_r";
        default:                                  return "initial radius (tag)";
        }
    }

    // Trails hold the previous palette, so they have to go with it.
    void cycle_color()
    {
        colorMode = (colorMode+1) % 3;
        clearFrames = clearCycle;
        std::cout << "colour: " << color_name(colorMode) << "\n";
    }

    Demo()
        : sim(syclQ, kNR, kNZ, kNPHI,
              float(kR0), float(kR), float(kLZ),
              /*U0=*/1.f, /*Re=*/400.f, /*dt=*/0.002f)
    {}

    bool init(SDL_MetalView sdlView)
    {
        layer = (CA::MetalLayer*)SDL_Metal_GetLayer(sdlView);

        std::cout << "SYCL device: "
                  << syclQ.get_device().get_info<sycl::info::device::name>() << "\n";

        part_px    = sycl::malloc_shared<float>(kNP,     syclQ);
        part_py    = sycl::malloc_shared<float>(kNP,     syclQ);
        part_pz    = sycl::malloc_shared<float>(kNP,     syclQ);
        color_buf  = sycl::malloc_shared<float>(kNP,     syclQ);
        render_buf = sycl::malloc_shared<float>(kNP * 4, syclQ);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> rr(kR0*1.01f, kR*0.99f);
        std::uniform_real_distribution<float> rphi(0.f, float(2*M_PI));
        std::uniform_real_distribution<float> rz(0.f, kLZ);
        for (int ip = 0; ip < kNP; ip++) {
            float pr   = rr(rng);
            float pphi = rphi(rng);
            part_px[ip]   = pr * std::cos(pphi);
            part_py[ip]   = pr * std::sin(pphi);
            part_pz[ip]   = rz(rng);
            // Lagrangian marker: where the particle started radially, so the
            // outflow jets visibly carry inner fluid to the outer wall.  A
            // particle that escapes and gets reseeded keeps its old marker,
            // so this mode slowly decorrelates -- fine for watching transport.
            color_buf[ip] = (pr - kR0) / (kR - kR0);
        }

#ifdef SYCL_EXT_ACPP_BACKEND_METAL
        // Events can only be shared with the device the SYCL queue actually runs
        // on, so Metal's device comes from SYCL rather than the other way round.
        if (syclQ.get_device().get_backend() == sycl::backend::metal) {
            dev = sycl::get_native<sycl::backend::metal>(syclQ.get_device());
            if (dev) { dev->retain(); interop = true; }   // balances release() in ~Demo
        }
#endif
        if (!dev) dev = MTL::CreateSystemDefaultDevice();
        if (!dev) { std::cerr << "No Metal device\n"; return false; }
        layer->setDevice(dev);
        layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
        layer->setFramebufferOnly(false);
        layer->setDisplaySyncEnabled(vsync);
        std::cout << "DIAG displaySyncEnabled=" << layer->displaySyncEnabled()
                  << " maxDrawables=" << layer->maximumDrawableCount() << "\n";
        clearCycle  = int(layer->maximumDrawableCount());
        if (clearCycle < 1) clearCycle = 3;
        clearFrames = clearCycle;   // start from a clean set of drawables
        renderQ = dev->newCommandQueue();

        if (interop) {
            renderDone = dev->newSharedEvent();
            if (!renderDone) interop = false;
        }
        std::cout << "compute/render sync: "
                  << (interop ? "Metal shared events (SYCL interop, GPU-side)"
                              : "CPU wait (fallback)") << "\n";

        // render_buf is USM, and the Metal backend keeps it inside a real
        // MTL::Buffer -- ask SYCL for that buffer and let the vertex shader read
        // the particles in place, instead of pushing them through a memcpy into
        // a private copy every frame.  The allocator sub-allocates, hence offset.
#ifdef SYCL_EXT_ACPP_BACKEND_METAL
        if (interop) {
            auto alloc = sycl::get_native_allocation<sycl::backend::metal>(
                render_buf, syclQ.get_context());
            // Shared storage means the buffer is the very host memory SYCL
            // handed out; if the two disagree the offset is not what we think
            // it is and the vertex shader would read the wrong particles.
            const bool sane = alloc.buffer &&
                (!alloc.buffer->contents() ||
                 static_cast<char*>(alloc.buffer->contents()) + alloc.offset ==
                     reinterpret_cast<char*>(render_buf));
            if (sane) {
                alloc.buffer->retain();          // balances release() in ~Demo
                renderMetalBuf  = alloc.buffer;
                renderBufOffset = NS::UInteger(alloc.offset);
                zeroCopy        = true;
            } else if (alloc.buffer) {
                std::cerr << "zero copy rejected: USM pointer does not match "
                             "the Metal buffer -- falling back to memcpy\n";
            }
        }
#endif
        if (!renderMetalBuf)
            renderMetalBuf = dev->newBuffer(kNP * 4 * sizeof(float),
                                            MTL::ResourceStorageModeShared);
        if (!renderMetalBuf) { std::cerr << "MTLBuffer alloc failed\n"; return false; }
        std::cout << "render buffer: "
                  << (zeroCopy ? "SYCL USM read in place (zero copy)"
                               : "separate buffer, memcpy per frame")
                  << "  offset=" << renderBufOffset << "\n";

        NS::Error* err = nullptr;
        auto* src = NS::String::string(kMSL, NS::UTF8StringEncoding);
        auto* lib = dev->newLibrary(src, nullptr, &err);
        if (!lib) {
            std::cerr << "Shader error: " << err->localizedDescription()->utf8String() << "\n";
            return false;
        }
        auto* fv  = lib->newFunction(NS::String::string("fade_vert", NS::UTF8StringEncoding));
        auto* ff2 = lib->newFunction(NS::String::string("fade_frag", NS::UTF8StringEncoding));
        auto* vf  = lib->newFunction(NS::String::string("ns_vert",   NS::UTF8StringEncoding));
        auto* ff  = lib->newFunction(NS::String::string("ns_frag",   NS::UTF8StringEncoding));
        lib->release();

        // Fade PSO
        auto* fpd = MTL::RenderPipelineDescriptor::alloc()->init();
        fpd->setVertexFunction(fv);
        fpd->setFragmentFunction(ff2);
        auto* fca = fpd->colorAttachments()->object(0);
        fca->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
        fca->setBlendingEnabled(true);
        fca->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
        fca->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        fca->setSourceAlphaBlendFactor(MTL::BlendFactorZero);
        fca->setDestinationAlphaBlendFactor(MTL::BlendFactorOne);
        fadePSO = dev->newRenderPipelineState(fpd, &err);
        fv->release(); ff2->release(); fpd->release();
        if (!fadePSO) {
            std::cerr << "Fade PSO error: " << err->localizedDescription()->utf8String() << "\n";
            return false;
        }

        // Particle PSO
        auto* pd = MTL::RenderPipelineDescriptor::alloc()->init();
        pd->setVertexFunction(vf);
        pd->setFragmentFunction(ff);
        auto* ca = pd->colorAttachments()->object(0);
        ca->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
        ca->setBlendingEnabled(true);
        ca->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
        ca->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        ca->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
        ca->setDestinationAlphaBlendFactor(MTL::BlendFactorZero);
        pso = dev->newRenderPipelineState(pd, &err);
        vf->release(); ff->release(); pd->release();
        if (!pso) {
            std::cerr << "PSO error: " << err->localizedDescription()->utf8String() << "\n";
            return false;
        }

        std::cout << "Grid: r=[" << kR0 << "," << kR << "]  phi=" << kNPHI
                  << "  z=" << kNZ << "  r=" << kNR
                  << "  Re=" << sim.Re << "  dt=" << sim.dt
                  << "  particles=" << kNP
                  << "  steps/frame=" << stepsPerFrame << "\n";
        std::cout << "Keys: arrows rotate, space pauses, C cycles colour, Esc quits\n"
                     "colour: " << color_name(colorMode) << "\n";
        return true;
    }

    void step()
    {
        if (showFps) report_fps();
        auto t_f0 = std::chrono::steady_clock::now();

        // Metal has to finish reading renderMetalBuf before SYCL overwrites it.
        // With interop that dependency lives on the GPU: the imported event is
        // enqueued into the in-order queue, so every kernel below waits for it.
        // Without interop the CPU has to block instead.
#ifdef SYCL_EXT_ACPP_BACKEND_METAL
        if (interop && renderDoneValue) {
            sycl::event rendered = sycl::make_event<sycl::backend::metal>(
                {renderDone, renderDoneValue}, syclQ.get_context());
            syclQ.submit([&](sycl::handler& cgh) {
                cgh.depends_on(rendered);
                cgh.single_task([]() {});
            });
        }
#endif
        if (!interop && prevCB) {
            prevCB->waitUntilCompleted(); prevCB->release(); prevCB = nullptr;
        }

        if (!paused)
            for (int k = 0; k < stepsPerFrame; k++) sim.step();
        sim.advect_particles(part_px, part_py, part_pz, color_buf, render_buf,
                             kNP, frame++, colorMode);
        // In-order queue: whatever is enqueued here runs after advect.  With
        // zero copy there is nothing left to transfer, so an empty task is
        // enqueued purely to give the render pass an event to wait for.
        sycl::event uploaded =
            zeroCopy ? syclQ.single_task([]() {})
                     : syclQ.memcpy(renderMetalBuf->contents(), render_buf,
                                    kNP * 4 * sizeof(float));
        if (!interop) syclQ.wait();
        t_sycl += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_f0).count();
        auto t_r0 = std::chrono::steady_clock::now();

        auto t_nd0 = std::chrono::steady_clock::now();
        CA::MetalDrawable* drawable = layer->nextDrawable();
        t_wait_drawable += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_nd0).count();
        if (!drawable) return;

        auto* rpd = MTL::RenderPassDescriptor::alloc()->init();
        auto* att = rpd->colorAttachments()->object(0);
        att->setTexture(drawable->texture());
        const bool wipe = clearFrames > 0;
        att->setLoadAction(wipe ? MTL::LoadActionClear : MTL::LoadActionLoad);
        att->setClearColor(MTL::ClearColor(0, 0, 0, 1));
        if (wipe) clearFrames--;
        att->setStoreAction(MTL::StoreActionStore);

        auto* cb  = renderQ->commandBuffer();
#ifdef SYCL_EXT_ACPP_BACKEND_METAL
        // Rendering waits for the SYCL upload on the GPU timeline.
        if (interop) {
            auto h = sycl::get_native<sycl::backend::metal>(uploaded);
            cb->encodeWait(h.event, h.value);
        }
#endif
        drawableW = int(drawable->texture()->width());
        drawableH = int(drawable->texture()->height());
        auto* enc = cb->renderCommandEncoder(rpd);

        // 1. Fade
        enc->setRenderPipelineState(fadePSO);
        float alpha = kTrailAlpha;
        enc->setFragmentBytes(&alpha, sizeof(alpha), NS::UInteger(0));
        enc->drawPrimitives(MTL::PrimitiveTypeTriangleStrip,
                            NS::UInteger(0), NS::UInteger(4));

        // 2. Particles
        enc->setRenderPipelineState(pso);
        enc->setVertexBuffer(renderMetalBuf, renderBufOffset, NS::UInteger(0));
        float rot[4] = {std::cos(angle_h), std::sin(angle_h),
                        std::cos(angle_v), std::sin(angle_v)};
        enc->setVertexBytes(rot, sizeof(rot), NS::UInteger(1));
        enc->setFragmentBytes(&colorMode, sizeof(colorMode), NS::UInteger(0));
        enc->drawPrimitives(MTL::PrimitiveTypePoint,
                            NS::UInteger(0), NS::UInteger(kNP));
        enc->endEncoding();
        rpd->release();

        cb->presentDrawable(drawable);
        if (interop) cb->encodeSignalEvent(renderDone, ++renderDoneValue);
        cb->retain();
        cb->commit();
        if (prevCB) prevCB->release();
        prevCB = cb;   // kept so the next frame (or ~Demo) can wait on this one
        t_render += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_r0).count();
    }

    void report_fps()
    {
        static auto t0 = std::chrono::steady_clock::now();
        static int  n  = 0;
        if (++n < 240) return;
        std::cout << "DIAG nextDrawable=" << (t_wait_drawable/n*1000)
                  << "  sycl=" << (t_sycl/n*1000)
                  << "  render=" << (t_render/n*1000) << " ms/frame  drawable="
                  << drawableW << "x" << drawableH << "\n";
        t_wait_drawable = t_sycl = t_render = 0;
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "fps: " << n/dt << std::endl;
        n  = 0;
        t0 = std::chrono::steady_clock::now();
    }

    ~Demo()
    {
        if (prevCB)           { prevCB->waitUntilCompleted(); prevCB->release(); }
        if (renderDone)       renderDone->release();
        if (pso)              pso->release();
        if (fadePSO)          fadePSO->release();
        if (renderMetalBuf)   renderMetalBuf->release();
        if (renderQ)          renderQ->release();
        if (dev)              dev->release();
        if (part_px)    sycl::free(part_px,    syclQ);
        if (part_py)    sycl::free(part_py,    syclQ);
        if (part_pz)    sycl::free(part_pz,    syclQ);
        if (color_buf)  sycl::free(color_buf,  syclQ);
        if (render_buf) sycl::free(render_buf, syclQ);
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
    bool vsync = true, showFps = false;
    int stepsPerFrame = 3;
    const auto parseSteps = [](std::string_view text, int& value) {
        int parsed = 0;
        const auto result = std::from_chars(
            text.data(), text.data()+text.size(), parsed);
        if (result.ec != std::errc() || result.ptr != text.data()+text.size()
            || parsed <= 0) {
            return false;
        }
        value = parsed;
        return true;
    };
    const auto usage = [&]() {
        std::cerr << "usage: " << argv[0]
                  << " [--no-vsync] [--fps] [--steps-per-frame=N]\n";
    };

    for (int i = 1; i < argc; i++) {
        const std::string_view arg = argv[i];
        constexpr std::string_view prefix = "--steps-per-frame=";
        if      (arg == "--no-vsync") vsync   = false;
        else if (arg == "--fps")      showFps = true;
        else if (arg.starts_with(prefix)) {
            if (!parseSteps(arg.substr(prefix.size()), stepsPerFrame)) {
                usage();
                return 1;
            }
        } else if (arg == "--steps-per-frame") {
            if (++i == argc || !parseSteps(argv[i], stepsPerFrame)) {
                usage();
                return 1;
            }
        } else {
            usage();
            return 1;
        }
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "NS Cylinder  ·  SYCL compute + Metal render",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        768, 768,
        SDL_WINDOW_METAL | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_RESIZABLE);
    if (!window) {
        std::cerr << "SDL_CreateWindow: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_MetalView metalView = SDL_Metal_CreateView(window);
    if (!metalView) {
        std::cerr << "SDL_Metal_CreateView failed\n";
        return 1;
    }

    Demo demo;
    demo.vsync   = vsync;
    demo.showFps = showFps;
    demo.stepsPerFrame = stepsPerFrame;
    if (!demo.init(metalView)) return 1;

    bool running = true;
    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) running = false;
            if (ev.type == SDL_KEYDOWN) {
                switch (ev.key.keysym.sym) {
                case SDLK_ESCAPE: running = false;               break;
                case SDLK_SPACE:  demo.paused = !demo.paused;   break;
                case SDLK_LEFT:   demo.rotate(-0.05f,  0.f);    break;
                case SDLK_RIGHT:  demo.rotate(+0.05f,  0.f);    break;
                case SDLK_UP:     demo.rotate( 0.f,   -0.05f);  break;
                case SDLK_DOWN:   demo.rotate( 0.f,   +0.05f);  break;
                case SDLK_c:      demo.cycle_color();          break;
                }
            }
        }
        demo.step();
    }

    SDL_Metal_DestroyView(metalView);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
