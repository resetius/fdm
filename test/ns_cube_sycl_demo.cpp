// ns_cube_sycl_demo.cpp
// Lid-driven 3-D Navier-Stokes cavity — SYCL compute + Metal visualization.
// New implementation; existing fdm code is not modified.
//
// NSCubeSycl<T>  — tensors backed by sycl::malloc_shared, kernels replace
//                  the OpenMP loops from ns_cube.cpp; LaplCube stays on CPU
//                  (unified memory on Apple Silicon, no copy needed).
// Demo           — SDL2 window + Metal: renders passive particles advected
//                  through the velocity field (XZ projection, jet colormap).

// ── metal-cpp (declarations only here; implementations in *_metal_impl.cpp) ──
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

// Foundation.hpp defines nil=nullptr, which conflicts with AdaptiveCpp internals
#ifdef nil
#  undef nil
#endif

// ── SDL2 ──────────────────────────────────────────────────────────────────────
#include <SDL2/SDL.h>
#include <SDL2/SDL_metal.h>

// ── SYCL + simulation ─────────────────────────────────────────────────────────
#include "ns_cube_sycl.h"

// ── Standard ──────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

// ═════════════════════════════════════════════════════════════════════════════
// Metal shaders
// ═════════════════════════════════════════════════════════════════════════════
// Trail fade: fraction of brightness lost per frame (~1 s at 60 fps)
static constexpr float kTrailAlpha = 0.015f;

static const char kMSL[] = R"msl(
#include <metal_stdlib>
using namespace metal;

// ── Trail fade: full-screen dark quad ────────────────────────────────────────
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
    float  psize [[point_size]];
};

static float3 hsv2rgb(float h)
{
    float3 rgb = clamp(abs(fmod(h*6.f + float3(0.f,4.f,2.f), 6.f) - 3.f) - 1.f, 0.f, 1.f);
    return mix(float3(1.f), rgb, 0.85f);
}

// pts = float4[]{x/hlx, y/hly, z/hlz, hue}
// rot = {cos_h, sin_h, cos_v, sin_v}
// 3D orthographic: Ry(h) then Rx(v), project onto XY screen
vertex VOut ns_vert(uint                 vid [[vertex_id]],
                    const device float4* pts [[buffer(0)]],
                    constant float4&     rot [[buffer(1)]])
{
    float3 p   = pts[vid].xyz;
    float  ch  = rot.x, sh = rot.y;
    float  cv  = rot.z, sv = rot.w;
    // rotate around Y by h, then around X by v
    float rx =  ch*p.x + sh*p.z;
    float ry =  sv*sh*p.x + cv*p.y - sv*ch*p.z;
    VOut o;
    o.pos   = float4(rx, ry, 0.f, 1.f);
    o.hue   = pts[vid].w;
    o.psize = 2.5f;
    return o;
}

fragment float4 ns_frag(VOut in [[stage_in]])
{
    return float4(hsv2rgb(in.hue), 1.f);
}
)msl";


// ═════════════════════════════════════════════════════════════════════════════
// Demo
// ═════════════════════════════════════════════════════════════════════════════
static constexpr int kNX=128, kNY=128, kNZ=128;
static constexpr int kNP=32768;

struct Demo {
    sycl::queue syclQ{
        []() {
            for (auto& plat : sycl::platform::get_platforms())
                for (auto& dev : plat.get_devices())
                    if (dev.is_gpu()) return dev;
            return sycl::device{sycl::cpu_selector_v};
        }(),
        sycl::property::queue::in_order{}};

    fdm::NSCubeSycl<float> sim;

    // Particle positions (physical space), per-particle hue, render buffer
    float *part_px=nullptr, *part_py=nullptr, *part_pz=nullptr;
    float *color_buf=nullptr;    // hue per particle in [0,1], fixed at init
    float *render_buf=nullptr;   // float4 per particle: {x/hlx, y/hly, z/hlz, hue}

    MTL::Device*              dev     = nullptr;
    MTL::CommandQueue*        renderQ = nullptr;
    MTL::RenderPipelineState* pso     = nullptr;
    MTL::RenderPipelineState* fadePSO = nullptr;
    MTL::Buffer*              partBuf = nullptr;  // Metal-readable copy of render_buf
    CA::MetalLayer*           layer   = nullptr;

    uint32_t frame      = 0;
    bool     firstFrame = true;
    bool     paused     = false;  // space toggles simulation
    float    angle_h    = 0.f;   // horizontal rotation (left/right arrows)
    float    angle_v    = 0.f;   // vertical tilt (up/down arrows)

    Demo()
        : sim(syclQ, kNX, kNY, kNZ,
              float(2*M_PI), float(2*M_PI), float(2*M_PI),
              /*U0=*/1.f, /*Re=*/40.f, /*dt=*/0.005f)
    {}

    bool init(SDL_MetalView sdlView)
    {
        layer = (CA::MetalLayer*)SDL_Metal_GetLayer(sdlView);

        std::cout << "SYCL device: "
                  << syclQ.get_device().get_info<sycl::info::device::name>() << "\n";

        // Allocate particle buffers in USM
        part_px    = sycl::malloc_shared<float>(kNP, syclQ);
        part_py    = sycl::malloc_shared<float>(kNP, syclQ);
        part_pz    = sycl::malloc_shared<float>(kNP, syclQ);
        color_buf  = sycl::malloc_shared<float>(kNP, syclQ);
        render_buf = sycl::malloc_shared<float>(kNP * 4, syclQ);

        // Seed particles randomly in the XZ visualization plane (py=0)
        const float hlx = (float)(sim.lx * 0.5f);
        const float hly = (float)(sim.ly * 0.5f);
        const float hlz = (float)(sim.lz * 0.5f);
        const float dx  = (float)sim.dx;
        const float dy  = (float)sim.dy;
        const float dz  = (float)sim.dz;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> rx(-(hlx - dx), hlx - dx);
        std::uniform_real_distribution<float> ry(-(hly - dy), hly - dy);
        std::uniform_real_distribution<float> rz(-(hlz - dz), hlz - dz);
        std::uniform_real_distribution<float> rc(0.f, 1.f);
        for (int ip = 0; ip < kNP; ip++) {
            part_px[ip]   = rx(rng);
            part_py[ip]   = ry(rng);
            part_pz[ip]   = rz(rng);
            color_buf[ip] = rc(rng);
        }

        dev = MTL::CreateSystemDefaultDevice();
        if (!dev) { std::cerr << "No Metal device\n"; return false; }
        layer->setDevice(dev);
        layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
        layer->setFramebufferOnly(false);     // needed to load previous frame for trails
        layer->setDisplaySyncEnabled(false);  // don't throttle on vsync
        renderQ = dev->newCommandQueue();

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

        // Fade PSO: full-screen quad that darkens previous frame (trail effect)
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

        // Metal shared buffer to receive render_buf each frame
        partBuf = dev->newBuffer(kNP * 4 * sizeof(float), MTL::ResourceStorageModeShared);
        if (!partBuf) { std::cerr << "MTLBuffer alloc failed\n"; return false; }

        std::cout << "Grid: " << kNX << "x" << kNY << "x" << kNZ
                  << "  Re=" << sim.Re << "  dt=" << sim.dt
                  << "  particles=" << kNP << "\n";
        return true;
    }

    void step()
    {
        if (!paused)
            for (int k = 0; k < 5; k++) sim.step();
        sim.advect_particles(part_px, part_py, part_pz, color_buf, render_buf, kNP, frame++);

        // Upload render_buf to Metal shared buffer
        std::memcpy(partBuf->contents(), render_buf, kNP * 4 * sizeof(float));

        // Render
        CA::MetalDrawable* drawable = layer->nextDrawable();
        if (!drawable) return;

        auto* rpd = MTL::RenderPassDescriptor::alloc()->init();
        auto* att = rpd->colorAttachments()->object(0);
        att->setTexture(drawable->texture());
        att->setLoadAction(firstFrame ? MTL::LoadActionClear : MTL::LoadActionLoad);
        att->setClearColor(MTL::ClearColor(0, 0, 0, 1));
        firstFrame = false;
        att->setStoreAction(MTL::StoreActionStore);

        auto* cb  = renderQ->commandBuffer();
        auto* enc = cb->renderCommandEncoder(rpd);

        // 1. Fade: darken previous frame to create trail decay
        enc->setRenderPipelineState(fadePSO);
        float alpha = kTrailAlpha;
        enc->setFragmentBytes(&alpha, sizeof(alpha), NS::UInteger(0));
        enc->drawPrimitives(MTL::PrimitiveTypeTriangleStrip,
                            NS::UInteger(0), NS::UInteger(4));

        // 2. Draw current particle positions
        enc->setRenderPipelineState(pso);
        enc->setVertexBuffer(partBuf, NS::UInteger(0), NS::UInteger(0));
        float rot[4] = {std::cos(angle_h), std::sin(angle_h),
                        std::cos(angle_v), std::sin(angle_v)};
        enc->setVertexBytes(rot, sizeof(rot), NS::UInteger(1));
        enc->drawPrimitives(MTL::PrimitiveTypePoint,
                            NS::UInteger(0), NS::UInteger(kNP));
        enc->endEncoding();
        rpd->release();

        cb->presentDrawable(drawable);
        cb->commit();
    }

    ~Demo()
    {
        if (pso)        pso->release();
        if (fadePSO)    fadePSO->release();
        if (partBuf)    partBuf->release();
        if (renderQ)    renderQ->release();
        if (dev)        dev->release();
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
int main()
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "NS Cube  ·  SYCL compute + Metal render",
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
    if (!demo.init(metalView)) return 1;

    bool running = true;
    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) running = false;
            if (ev.type == SDL_KEYDOWN) {
                switch (ev.key.keysym.sym) {
                case SDLK_ESCAPE: running = false;          break;
                case SDLK_SPACE:  demo.paused = !demo.paused; break;
                case SDLK_LEFT:   demo.angle_h -= 0.05f;     break;
                case SDLK_RIGHT:  demo.angle_h += 0.05f;     break;
                case SDLK_UP:     demo.angle_v -= 0.05f;     break;
                case SDLK_DOWN:   demo.angle_v += 0.05f;     break;
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
