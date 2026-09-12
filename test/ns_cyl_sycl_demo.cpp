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
#include "ns_cyl_spectral_filter_sycl.h"
#include "ns_cyl_spectral_storage.h"
#include "ns_cyl_state.h"
#include "sycl_queue_properties.h"

// ── Standard ──────────────────────────────────────────────────────────────────
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <chrono>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// ═════════════════════════════════════════════════════════════════════════════
// Metal shaders
// ═════════════════════════════════════════════════════════════════════════════
// How hard each frame pulls the target toward the background: smaller means
// longer trails.  The light theme needs a gentler pull, because ink on paper
// lifts faster than glow on black.
static constexpr float kTrailAlphaDark  = 0.015f;
static constexpr float kTrailAlphaLight = 0.010f;

// Background in LINEAR sRGB: the target is an sRGB format, so the hardware
// applies the transfer function on write.  Paper white here is sRGB ~0.97.
static constexpr float kGroundDark[3]  = {0.000f, 0.000f, 0.000f};
static constexpr float kGroundLight[3] = {0.937f, 0.933f, 0.918f};

static const char kMSL[] = R"msl(
#include <metal_stdlib>
using namespace metal;

// ── Trail fade ────────────────────────────────────────────────────────────────
vertex float4 fade_vert(uint vid [[vertex_id]])
{
    float2 pos[4] = {float2(-1,1), float2(1,1), float2(-1,-1), float2(1,-1)};
    return float4(pos[vid], 0, 1);
}
// Trails are accumulated raster: every frame pulls the target a little toward
// one colour, and that colour IS the background.  Fading toward black on a
// light theme would grey the whole picture out, so it is passed in.
fragment float4 fade_frag(constant float4& fade [[buffer(0)]])
{
    return fade;
}

// ── Oklab ─────────────────────────────────────────────────────────────────────
// Perceptually uniform, unlike HSV/HSL: a hue sweep holds its lightness instead
// of flaring at yellow and sinking at blue, and a diverging ramp changes only
// what it is meant to.  Matrices from Ottosson's Oklab note.  Both ends of the
// conversion are LINEAR sRGB -- which is exactly what the sRGB render target
// wants, since the hardware does the transfer function on write.
static float3 oklab_to_linear_srgb(float3 c)
{
    float l_ = c.x + 0.3963377774f*c.y + 0.2158037573f*c.z;
    float m_ = c.x - 0.1055613458f*c.y - 0.0638541728f*c.z;
    float s_ = c.x - 0.0894841775f*c.y - 1.2914855480f*c.z;
    float l = l_*l_*l_, m = m_*m_*m_, s = s_*s_*s_;
    return float3(
        +4.0767416621f*l - 3.3077115913f*m + 0.2309699292f*s,
        -1.2684380046f*l + 2.6097574011f*m - 0.3413193965f*s,
        -0.0041960863f*l - 0.7034186147f*m + 1.7076147010f*s);
}

// Lightness L, chroma C, hue h in turns.
static float3 oklch(float L, float C, float h)
{
    return oklab_to_linear_srgb(
        float3(L, C*cos(h*6.28318531f), C*sin(h*6.28318531f)));
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
// Ink lifts off paper far faster than glow leaves black, so the far half must
// not be faded nearly as hard or it disappears into the ground.
constant float kDepthFadeLight = 0.32f;
// Fewer particles carrying fatter marks read better than a fine mist: the
// strokes overlap into continuous ribbons instead of dithering.
constant float kSizeFar   = 2.2f;
constant float kSizeNear  = 5.5f;

// theme 0 = dark ground, 1 = light ground.  Both palettes live in Oklab, so
// the two themes differ only in lightness, not in which colours are used.

// Lagrangian marker: an unordered label, so a cyclic ramp at constant L and C.
// No hue then reads as "more" than another -- the failure of an HSV rainbow,
// where yellow screams and blue sinks although both claim value 1.
static float3 palette_tag(float t, int theme)
{
    return (theme == 0) ? oklch(0.80f, 0.13f, t)
                        : oklch(0.55f, 0.16f, t);
}

// Signed quantity, 0.5 = at rest: a straight line through a near-neutral in
// Oklab.  Only the sign carries colour, and still fluid sinks into the ground
// instead of competing with the structure.
// The light theme is ink on paper, so its neutral is a mid grey, not a
// near-white: paper sits at L ~ 0.98, and a neutral anywhere near it makes
// quiet fluid -- most of the frame -- vanish.  Speed then reads as ink
// density, with colour carrying only the sign.
static float3 palette_signed(float t, int theme)
{
    float  s       = clamp(t*2.f - 1.f, -1.f, 1.f);
    float3 neutral = (theme == 0) ? float3(0.62f,  0.0000f,  0.0050f)
                                  : float3(0.72f,  0.0000f,  0.0080f);
    float3 down    = (theme == 0) ? float3(0.70f, -0.0513f, -0.1410f)
                                  : float3(0.45f, -0.0581f, -0.1598f);
    float3 up      = (theme == 0) ? float3(0.74f,  0.1189f,  0.1070f)
                                  : float3(0.50f,  0.1338f,  0.1204f);
    return oklab_to_linear_srgb(
        mix(neutral, s < 0.f ? down : up, abs(s)));
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

// cfg = {colour mode, theme}
fragment float4 ns_frag(VOut in [[stage_in]],
                        constant int2& cfg [[buffer(0)]])
{
    // Squared so the falloff is concentrated on the far half: the front stays
    // at full strength while the back recedes into the trails behind it.  On a
    // light ground the same fade reads as aerial perspective.
    float far   = (cfg.y == 0) ? kDepthFade : kDepthFadeLight;
    float alpha = mix(far, 1.f, in.depth*in.depth);
    // Mode 0 is a Lagrangian marker -- an unordered label, so a cyclic ramp.
    // Modes 1 and 2 are signed velocities and need a diverging one.
    float3 rgb  = (cfg.x == 0) ? palette_tag(in.hue, cfg.y)
                               : palette_signed(in.hue, cfg.y);
    return float4(rgb, alpha);
}

// ── Reference frame ───────────────────────────────────────────────────────────
struct LOut {
    float4 pos [[position]];
    float  key;
    float  depth;
};

// Same rotation as ns_vert, so the frame tracks the flow exactly.
// seg = float4{x, y_vertical, z, key}; key picks the colour.
vertex LOut box_vert(uint                 vid [[vertex_id]],
                     const device float4* seg [[buffer(0)]],
                     constant float4&     rot [[buffer(1)]])
{
    float3 p  = seg[vid].xyz;
    float ch  = rot.x, sh = rot.y;
    float cv  = rot.z, sv = rot.w;
    float rx  =  ch*p.x + sh*p.z;
    float ry  =  sv*sh*p.x + cv*p.y - sv*ch*p.z;
    float rz  =  sv*p.y + cv*(ch*p.z - sh*p.x);

    LOut o;
    o.pos   = float4(rx, ry, 0.f, 1.f);
    o.depth = clamp(0.5f + 0.5f*rz/kZoom, 0.f, 1.f);
    o.key = seg[vid].w;
    return o;
}

// cfg = {unused, theme}
fragment float4 box_frag(LOut in [[stage_in]],
                         constant int2& cfg [[buffer(0)]])
{
    // One lightness for the whole frame, hues picked in Oklch so the three
    // axes are equally prominent -- the frame must not outshout the flow.
    // Kept lighter than the flow's neutral so the frame never outweighs it.
    const float L = (cfg.y == 0) ? 0.62f : 0.60f;
    const float k = in.key;
    float3 rgb = (k < 0.5f) ? oklch(L, 0.010f, 0.00f)    // box and ticks
               : (k < 1.5f) ? oklch(L, 0.105f, 0.07f)    // x
               : (k < 2.5f) ? oklch(L, 0.105f, 0.40f)    // z, the cylinder axis
               : (k < 3.5f) ? oklch(L, 0.105f, 0.72f)    // y
                            : oklch(L, 0.045f, 0.22f);   // annulus walls
    // Same depth cue as the particles, so near edges read in front of far ones.
    return float4(rgb, mix(0.20f, 0.65f, in.depth));
}
)msl";

// ═════════════════════════════════════════════════════════════════════════════
// Demo
// ═════════════════════════════════════════════════════════════════════════════
//static constexpr int kNR=32, kNZ=32, kNPHI=32;
static constexpr int kNR=64, kNZ=64, kNPHI=64;
//static constexpr int kNR=128, kNZ=128, kNPHI=128;

// Fewer, fatter marks: strokes overlap into continuous ribbons rather than
// dithering into a mist, and the vortex cores stay legible while moving.
static constexpr int kNP=12288;

// Initial perturbation amplitude relative to the inner-wall speed.
//static constexpr float kSeed = 1e-3f;
static constexpr float kSeed = 1e-2f;

static constexpr float kR0   = 1.5707963267948966;   // inner cylinder radius
static constexpr float kR    = 3.141592653589793;   // outer cylinder radius
static constexpr float kU0   = 1.0f;
static constexpr float kRe   = 100.0f;
static constexpr float kDefaultDt = 0.002f;

// Axial period.  A Taylor vortex is nearly square in cross-section, so its
// height is about the gap width d = kR - kR0 = 1.  z is periodic, so only
// whole wavelengths fit and one wavelength holds a counter-rotating pair:
// the vortex count is kLZ/d rounded to an even number.  Formally it is
// 2*round(k_c*kLZ/2pi) with the critical Taylor wavenumber k_c*d ~ 3.16.
// The flow starts from the Couette base plus broadband noise, and the fastest
// growing mode wins over the rest -- so the count below is what you get.
static constexpr float kLZ = 10.0f;
//static constexpr float kLZ = 2.0f;            //  2 vortices (k=3.14, best fit)
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

// ── Reference frame geometry ──────────────────────────────────────────────────
// Line list in the same render coordinates advect_particles() writes:
// {x/kR, z/kLZ*2-1, y/kR} scaled by kZoom.  So the bounding cube is exactly
// [-kZoom,kZoom]^3: its horizontal span is the outer diameter 2*kR and its
// vertical span the axial period kLZ.
static constexpr float kFrameZoom = 0.8f;   // must match kZoom in the shader
static constexpr int   kFrameTicks = 8;     // subdivisions per axis

static std::vector<float> build_frame_vertices()
{
    std::vector<float> v;
    auto line = [&](float ax, float ay, float az,
                    float bx, float by, float bz, float key) {
        v.insert(v.end(), {ax, ay, az, key, bx, by, bz, key});
    };

    const float z = kFrameZoom;

    // Twelve edges of the bounding cube.  The three meeting at the near-bottom
    // corner carry the axis colours, the rest stay grey.
    for (int i = 0; i < 4; ++i) {
        const float a = (i & 1) ? z : -z;
        const float b = (i & 2) ? z : -z;
        line(-z, a, b,  z, a, b, (i == 0) ? 1.f : 0.f);   // along x
        line(a, -z, b,  a, z, b, (i == 0) ? 2.f : 0.f);   // along z (vertical)
        line(a, b, -z,  a, b, z, (i == 0) ? 3.f : 0.f);   // along y
    }

    // Ticks on those three edges.  Length is a fixed fraction of the box so
    // they stay legible at any zoom.
    const float t = 0.035f*z;
    for (int i = 0; i <= kFrameTicks; ++i) {
        const float s = -z + 2*z*float(i)/kFrameTicks;
        const bool major = (i % (kFrameTicks/2) == 0);
        const float len = major ? 2*t : t;
        line(s, -z, -z,  s, -z-len, -z, 1.f);             // x axis
        line(-z, s, -z,  -z-len, s, -z, 2.f);             // z axis
        line(-z, -z, s,  -z-len, -z, s, 3.f);             // y axis
    }

    // The annulus itself: inner and outer wall circles at both ends, so the
    // domain is readable rather than guessed from the bounding box.
    const float inner = z*kR0/kR;
    constexpr int kSegments = 64;
    for (int end = 0; end < 2; ++end) {
        const float y = end ? z : -z;
        for (int i = 0; i < kSegments; ++i) {
            const float a0 = float(2*M_PI)*i/kSegments;
            const float a1 = float(2*M_PI)*(i+1)/kSegments;
            line(z*std::cos(a0), y, z*std::sin(a0),
                 z*std::cos(a1), y, z*std::sin(a1), 4.f);
            line(inner*std::cos(a0), y, inner*std::sin(a0),
                 inner*std::cos(a1), y, inner*std::sin(a1), 4.f);
        }
    }
    return v;
}

struct ProjectorInput {
    std::string filename;
    fdm::NSCylSpectralMetadata metadata;
    fdm::NSCylSpectralProjector<float> projector;
};

static void require_projector_value(
    const char* name, double actual, double expected)
{
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    const double tolerance =
        4*static_cast<double>(std::numeric_limits<float>::epsilon())*scale;
    if (std::abs(actual-expected) > tolerance) {
        throw std::runtime_error(
            std::string("projector ")+name+" mismatch: file="
            +std::to_string(actual)+", demo="+std::to_string(expected));
    }
}

static ProjectorInput load_projector(const std::string& filename)
{
    fdm::NSCylSpectralModeSet<float> modes;
    fdm::NSCylSpectralMetadata metadata;
    fdm::NSCylSpectralStorage(filename).load(modes, metadata);

    if (metadata.nr != kNR || metadata.nphi != kNPHI
        || metadata.nz != kNZ) {
        throw std::runtime_error(
            "projector grid mismatch: file="+std::to_string(metadata.nr)
            +"x"+std::to_string(metadata.nphi)+"x"
            +std::to_string(metadata.nz)+", demo="+std::to_string(kNR)
            +"x"+std::to_string(kNPHI)+"x"+std::to_string(kNZ));
    }
    require_projector_value("r0", metadata.r, kR0);
    require_projector_value("R", metadata.R, kR);
    require_projector_value("z0", metadata.h1, 0.0);
    require_projector_value("Lz", metadata.h2-metadata.h1, kLZ);
    require_projector_value("Re", metadata.reynolds, kRe);
    require_projector_value("U0", metadata.wall_speed, kU0);

    return {
        filename,
        metadata,
        fdm::NSCylSpectralProjector<float>(
            modes, metadata.condition_limit)
    };
}

struct Demo {
    sycl::queue syclQ{
        []() {
            for (auto& plat : sycl::platform::get_platforms())
                for (auto& dev : plat.get_devices())
                    if (dev.is_gpu()) return dev;
            return sycl::device{sycl::cpu_selector_v};
        }(),
        fdm::sycl_in_order_queue_properties()};

    fdm::NSCylSycl<float> sim;
    fdm::NSCylStateLayout<float> stateLayout;
    std::unique_ptr<fdm::NSCylSpectralFilterSycl<float>> spectralFilter;
    std::vector<float> couetteReference;
    std::string projectorFilename;

    float *part_px=nullptr, *part_py=nullptr, *part_pz=nullptr;
    float *color_buf=nullptr;
    float *render_buf=nullptr;   // float4 per particle: {x/R, y/R, z_norm, hue}

    MTL::Device*              dev     = nullptr;
    MTL::CommandQueue*        renderQ = nullptr;
    MTL::RenderPipelineState* pso     = nullptr;
    MTL::RenderPipelineState* fadePSO = nullptr;
    MTL::RenderPipelineState* boxPSO  = nullptr;
    MTL::Buffer*              frameBuf = nullptr;   // reference-frame line list
    int                       frameVertices = 0;
    bool                      showFrame = true;
    int                       theme = 1;   // 0 = dark ground, 1 = light
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
    uint64_t simulationSteps = 0;
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

    // The wipe matters here: trails already in the target were laid down over
    // the other ground, and fading them toward the new one leaves a stain.
    void toggle_theme()
    {
        theme = !theme;
        clearFrames = clearCycle;
        std::cout << "ground: " << (theme ? "light" : "dark") << "\n";
    }

    // Restore the initial particle distribution without changing the flow.
    void reset_particles()
    {
        syclQ.wait(); // The previous frame may still be reading the particles.
        clear_trails();

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
        clearFrames = clearCycle;
    }

    explicit Demo(std::optional<ProjectorInput> projectorInput)
        : sim(syclQ, kNR, kNZ, kNPHI,
              float(kR0), float(kR), float(kLZ),
              kU0, kRe, kDefaultDt)
        , stateLayout(kNR, kNZ, kNPHI)
    {
        init_couette_base();
        couetteReference = pack_state();
        seed_noise();
        if (projectorInput) {
            projectorFilename = std::move(projectorInput->filename);
            const int dimension = projectorInput->projector.real_dimension();
            const int blocks = static_cast<int>(
                projectorInput->projector.blocks().size());
            spectralFilter =
                std::make_unique<fdm::NSCylSpectralFilterSycl<float>>(
                    syclQ, sim.nr, sim.nphi, sim.nz,
                    std::move(projectorInput->projector));
            std::cout << "projector: " << projectorFilename
                      << "  blocks=" << blocks
                      << "  real_dimension=" << dimension
                      << "  stored_steps="
                      << projectorInput->metadata.operator_steps
                      << "  stored_dt=" << projectorInput->metadata.dt
                      << "  demo_dt=" << sim.dt << "\n";
        }
    }

    // ── Particle history ──────────────────────────────────────────────────────
    // The on-screen trails are accumulated raster: each frame fades what is
    // already in the target, so nothing about them survives into a snapshot.
    // Vector trajectories therefore need the positions kept on the host.  Only
    // a subset is recorded -- every trail is a polyline in the EPS, and all
    // kNP of them would make a file nobody can open.
    // A trail has to span a vortex turnover to show a vortex, and at
    // stepsPerFrame*dt per frame that is hundreds of frames.  Sampling every
    // kTrailEvery-th frame buys that span without paying for it in points:
    // the covered time is kTrailSamples*kTrailEvery*stepsPerFrame*dt, here
    // about 11.5 time units.  Raising kTrailSamples lengthens the trail at the
    // same resolution; raising kTrailEvery lengthens it for free but makes the
    // polyline more angular, which shows on a curving vortex path.
    static constexpr int kTrailParticles = 512;
    static constexpr int kTrailSamples   = 8192;
    static constexpr int kTrailEvery     = 10;
    static constexpr int kTrailStride    = kNP/kTrailParticles;

    std::vector<float> trailHist;   // float4 per (sample, particle), ring buffer
    int trailHead = 0;              // next slot to write
    int trailCount = 0;             // valid samples, <= kTrailSamples
    int trailPhase = 0;             // frames since the last recorded sample

    void record_trails()
    {
        if (trailPhase++ % kTrailEvery) { return; }
        if (trailHist.empty()) {
            trailHist.assign(std::size_t(kTrailSamples)*kTrailParticles*4, 0.f);
        }
        float* slot = trailHist.data()
                    + std::size_t(trailHead)*kTrailParticles*4;
        for (int i = 0; i < kTrailParticles; ++i) {
            std::memcpy(slot+4*i, render_buf+4*(i*kTrailStride),
                        4*sizeof(float));
        }
        trailHead = (trailHead+1)%kTrailSamples;
        if (trailCount < kTrailSamples) { ++trailCount; }
    }

    void clear_trails() { trailHead = 0; trailCount = 0; trailPhase = 0; }

    // ── EPS snapshot ──────────────────────────────────────────────────────────
    // Metal is a rasteriser and cannot hand back vectors, but the scene is just
    // points and lines under a known orthographic projection, so the figure is
    // written directly.  Output is true vector art at any scale -- the format
    // the journal template uses.  Two differences from the screen: particles are
    // painted far-to-near instead of alpha-blended unsorted, and the frame is
    // square, so the window aspect no longer stretches the picture.
    // Screen and page want opposite things here.  The display fades distant
    // particles so 32768 of them do not read as a wall; the figure carries a
    // thirtieth of that, and dimming strokes there only hides the vortices.
    static constexpr double kEpsDepthFade  = 1.0;    // 1 = no fade at all
    static constexpr double kEpsTrailShade = 0.80;

    void dump_eps()
    {
        static int counter = 0;
        const std::string name =
            "ns_cyl_frame_"+std::to_string(counter++)+".eps";
        std::ofstream out(name);
        if (!out) { std::cerr << "cannot write " << name << "\n"; return; }

        constexpr double kSide = 420.0;      // points, square
        constexpr double kMargin = 4.0;
        const double half = 0.5*kSide;
        auto sx = [&](double x) { return kMargin+half+half*x; };

        const float ch = std::cos(angle_h), sh = std::sin(angle_h);
        const float cv = std::cos(angle_v), sv = std::sin(angle_v);
        auto project = [&](float x, float y, float z,
                           double& px, double& py, double& depth) {
            px = ch*x + sh*z;
            py = sv*sh*x + cv*y - sv*ch*z;
            const double rz = sv*y + cv*(ch*z - sh*x);
            depth = std::clamp(0.5+0.5*rz/kFrameZoom, 0.0, 1.0);
        };

        out << "%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 "
            << int(kSide+2*kMargin) << " " << int(kSide+2*kMargin)
            << "\n%%Creator: ns_cyl_sycl_demo\n%%EndComments\n";
        // PostScript has no alpha, and the scene composites over black, so
        // every colour is premultiplied by its alpha instead.
        out << "0 0 0 setrgbcolor 0 0 " << kSide+2*kMargin << " "
            << kSide+2*kMargin << " rectfill\n";
        out << "/d { 0 360 arc fill } bind def\n"
               "/l { moveto lineto stroke } bind def\n0.4 setlinewidth\n";

        if (showFrame) {
            const std::vector<float> frame = build_frame_vertices();
            for (std::size_t i = 0; i+7 < frame.size(); i += 8) {
                double ax, ay, ad, bx, by, bd;
                project(frame[i+0], frame[i+1], frame[i+2], ax, ay, ad);
                project(frame[i+4], frame[i+5], frame[i+6], bx, by, bd);
                const float key = frame[i+3];
                double r = 0.42, g = 0.44, b = 0.50;
                if      (key > 3.5f) { r = 0.55; g = 0.50; b = 0.30; }
                else if (key > 2.5f) { r = 0.36; g = 0.56; b = 0.95; }
                else if (key > 1.5f) { r = 0.40; g = 0.80; b = 0.42; }
                else if (key > 0.5f) { r = 0.85; g = 0.32; b = 0.28; }
                const double a = kEpsDepthFade*(0.55+0.45*0.5*(ad+bd));
                out << r*a << " " << g*a << " " << b*a << " setrgbcolor "
                    << sx(ax) << " " << sx(ay) << " "
                    << sx(bx) << " " << sx(by) << " l\n";
            }
        }

        // Trajectories.  Particles teleport when they wrap in z or get reseeded
        // after leaving the annulus, so a jump longer than kBreak ends the
        // polyline instead of drawing a streak across the box.
        if (trailCount > 1) {
            constexpr double kBreak = 0.15;
            out << "0.35 setlinewidth\n";
            for (int p = 0; p < kTrailParticles; ++p) {
                const int newest = (trailHead-1+kTrailSamples)%kTrailSamples;
                const float* head = trailHist.data()
                                  + std::size_t(newest)*kTrailParticles*4+4*p;
                double r, g, b;
                eps_colour(head[3], r, g, b);
                out << r*kEpsTrailShade << " " << g*kEpsTrailShade << " "
                    << b*kEpsTrailShade << " setrgbcolor\n";

                bool open = false;
                double prevx = 0, prevy = 0;
                for (int k = 0; k < trailCount; ++k) {
                    const int f =
                        (trailHead-trailCount+k+2*kTrailSamples)%kTrailSamples;
                    const float* s = trailHist.data()
                                   + std::size_t(f)*kTrailParticles*4+4*p;
                    double x, y, d;
                    project(s[0], s[1], s[2], x, y, d);
                    const double jump = open
                        ? std::hypot(x-prevx, y-prevy) : 0.0;
                    if (open && jump > kBreak) {
                        out << "stroke\n";
                        open = false;
                    }
                    if (!open) {
                        out << "newpath " << sx(x) << " " << sx(y) << " moveto\n";
                        open = true;
                    } else {
                        out << sx(x) << " " << sx(y) << " lineto\n";
                    }
                    prevx = x; prevy = y;
                }
                if (open) { out << "stroke\n"; }
            }
            out << "0.4 setlinewidth\n";
        }

        // Heads of the same tracers, and only those: all kNP dots would bury
        // the trajectories they are supposed to head.
        // Painter's algorithm: far particles first, so the near ones cover them.
        std::vector<int> order(kTrailParticles);
        std::vector<double> depth(kTrailParticles);
        std::vector<double> px(kTrailParticles), py(kTrailParticles);
        for (int p = 0; p < kTrailParticles; ++p) {
            const int i = p*kTrailStride;
            project(render_buf[4*i+0], render_buf[4*i+1], render_buf[4*i+2],
                    px[p], py[p], depth[p]);
            order[p] = p;
        }
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return depth[a] < depth[b]; });

        for (int p : order) {
            const double d = depth[p];
            const double a = kEpsDepthFade;
            const double size = 1.4+(3.2-1.4)*d;
            double r, g, b;
            eps_colour(render_buf[4*(p*kTrailStride)+3], r, g, b);
            out << r*a << " " << g*a << " " << b*a << " setrgbcolor "
                << sx(px[p]) << " " << sx(py[p]) << " "
                << 0.5*size*kSide/700.0 << " d\n";
        }
        out << "showpage\n%%EOF\n";
        std::cout << "wrote " << name << "  ("
                  << (showFrame ? "with" : "without") << " frame, "
                  << color_name(colorMode) << ")\n";
    }

    // Host copies of the shader palettes, so the figure matches the screen.
    void eps_colour(float h, double& r, double& g, double& b) const
    {
        if (colorMode == fdm::NSCylSycl<float>::color_tag) {
            auto ch = [&](float o) {
                return double(std::clamp(
                    std::fabs(std::fmod(h*6.f+o, 6.f)-3.f)-1.f, 0.f, 1.f));
            };
            r = 1.0+0.85*(ch(0.f)-1.0);
            g = 1.0+0.85*(ch(4.f)-1.0);
            b = 1.0+0.85*(ch(2.f)-1.0);
        } else {
            const double s = std::clamp(double(h)*2.0-1.0, -1.0, 1.0);
            const double t = std::fabs(s);
            const double dr = s < 0 ? 0.15 : 1.00;
            const double dg = s < 0 ? 0.50 : 0.38;
            const double db = s < 0 ? 1.00 : 0.14;
            r = 0.50+(dr-0.50)*t;
            g = 0.52+(dg-0.52)*t;
            b = 0.58+(db-0.58)*t;
        }
    }

    std::vector<float> pack_state() const
    {
        std::vector<float> packed(stateLayout.state_size);
        auto u = sim.ua();
        auto v = sim.va();
        auto w = sim.wa();
        auto p = sim.pa();

        int index = stateLayout.u_offset;
        for (int i = 0; i < sim.nphi; i++)
            for (int k = 0; k < sim.nz; k++)
                for (int j = 1; j < sim.nr; j++)
                    packed[index++] = u(i,k,j);
        for (auto field : {v, w, p})
            for (int i = 0; i < sim.nphi; i++)
                for (int k = 0; k < sim.nz; k++)
                    for (int j = 1; j <= sim.nr; j++)
                        packed[index++] = field(i,k,j);
        return packed;
    }

    void unpack_state(const std::vector<float>& packed)
    {
        if (static_cast<int>(packed.size()) != stateLayout.state_size)
            throw std::invalid_argument("packed demo state has the wrong size");

        auto u = sim.ua();
        auto v = sim.va();
        auto w = sim.wa();
        auto p = sim.pa();
        int index = stateLayout.u_offset;
        for (int i = 0; i < sim.nphi; i++) {
            for (int k = 0; k < sim.nz; k++) {
                u(i,k,0) = 0;
                for (int j = 1; j < sim.nr; j++)
                    u(i,k,j) = packed[index++];
                u(i,k,sim.nr) = 0;
            }
        }
        for (auto field : {v, w, p})
            for (int i = 0; i < sim.nphi; i++)
                for (int k = 0; k < sim.nz; k++)
                    for (int j = 1; j <= sim.nr; j++)
                        field(i,k,j) = packed[index++];
        sim.apply_boundary_conditions();
    }

    void apply_spectral_filter()
    {
        if (!spectralFilter) {
            std::cout << "filter: no projector loaded; use --projector FILE.nc\n";
            return;
        }

        const auto diagnostics = spectralFilter->remove(
            sim, couetteReference);
        clearFrames = clearCycle;

        std::cout << "filter: t=" << simulationSteps*sim.dt
                  << "  blocks=" << diagnostics.blocks.size()
                  << "  velocity=" << diagnostics.velocity_perturbation_norm
                  << " -> " << diagnostics.filtered_velocity_norm
                  << "  removed_velocity="
                  << diagnostics.removed_velocity_norm
                  << "  remaining_unstable="
                  << diagnostics.remaining_unstable_norm << "\n";
    }

    // Add deterministic broadband noise; the first step projects it onto the
    // divergence-free subspace and enforces the wall conditions.
    void seed_noise()
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> noise(-kSeed*sim.U0, kSeed*sim.U0);

        auto u = sim.ua();
        auto v = sim.va();
        auto w = sim.wa();
        for (int i = 0; i < sim.nphi; i++) {
            for (int k = 0; k < sim.nz; k++) {
                // Radial velocity is staggered on faces; leave both walls set
                // by the Couette initializer.
                for (int j = 1; j < sim.nr; j++) {
                    u(i,k,j) += noise(rng);
                }
                for (int j = 1; j <= sim.nr; j++) {
                    v(i,k,j) += noise(rng);
                    w(i,k,j) += noise(rng);
                }
            }
        }
    }

    // Use the same stationary discrete Couette state as the spectral probe.
    // Sampling A*r+B/r at cell centres leaves a small viscous residual in the
    // staggered stencil and introduces an otherwise unrelated spin-up.
    //
    // Unlike the probe we keep U0 as it is: the probe zeroes it because
    // L_step advances a perturbation with homogeneous wall conditions, while
    // here the moving wall is part of the flow being integrated.
    void init_couette_base()
    {
        const auto velocity = fdm::make_discrete_couette_velocity<float>(sim);
        const auto pressure = fdm::make_discrete_couette_pressure(sim, velocity);

        auto w = sim.wa();
        auto p = sim.pa();
        for (int i = 0; i < sim.nphi; i++) {
            for (int k = 0; k < sim.nz; k++) {
                for (int j = 0; j <= sim.nr+1; j++) {
                    w(i,k,j) = velocity[j];
                    p(i,k,j) = pressure[j];
                }
            }
        }
        // u and v remain zero; p already balances the centrifugal term, so
        // there is no artificial pressure transient on the first step.
    }

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

        reset_particles();

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
        auto* bv  = lib->newFunction(NS::String::string("box_vert",  NS::UTF8StringEncoding));
        auto* bf  = lib->newFunction(NS::String::string("box_frag",  NS::UTF8StringEncoding));
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

        // Reference frame PSO, same blending as the particles.
        auto* bd = MTL::RenderPipelineDescriptor::alloc()->init();
        bd->setVertexFunction(bv);
        bd->setFragmentFunction(bf);
        auto* bca = bd->colorAttachments()->object(0);
        bca->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
        bca->setBlendingEnabled(true);
        bca->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
        bca->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        bca->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
        bca->setDestinationAlphaBlendFactor(MTL::BlendFactorZero);
        boxPSO = dev->newRenderPipelineState(bd, &err);
        bv->release(); bf->release(); bd->release();
        if (!boxPSO) {
            std::cerr << "Box PSO error: " << err->localizedDescription()->utf8String() << "\n";
            return false;
        }

        const std::vector<float> frame = build_frame_vertices();
        frameVertices = int(frame.size()/4);
        frameBuf = dev->newBuffer(frame.data(), frame.size()*sizeof(float),
                                  MTL::ResourceStorageModeShared);

        std::cout << "Grid: r=[" << kR0 << "," << kR << "]  phi=" << kNPHI
                  << "  z=" << kNZ << "  r=" << kNR
                  << "  Re=" << sim.Re << "  dt=" << sim.dt
                  << "  particles=" << kNP
                  << "  steps/frame=" << stepsPerFrame << "\n";
        std::cout << "Initial state: stationary discrete Couette base "
                     "(same base as the spectral probe) + noise "
                  << kSeed << "*U0\n";
        std::cout << "Keys: arrows rotate, space pauses, C cycles colour,\n"
                     "      F applies spectral filter, R reseeds particles,\n"
                     "      G toggles the reference frame, B the background,\n"
                     "      P writes an EPS, Esc quits\n"
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

        if (!paused) {
            for (int k = 0; k < stepsPerFrame; k++) sim.step();
            simulationSteps += stepsPerFrame;
        }
        sim.advect_particles(part_px, part_py, part_pz, color_buf, render_buf,
                             kNP, frame++, colorMode);
        record_trails();
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
        const float* ground = theme ? kGroundLight : kGroundDark;
        att->setClearColor(
            MTL::ClearColor(ground[0], ground[1], ground[2], 1));
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
        const float fade[4] = {ground[0], ground[1], ground[2],
                               theme ? kTrailAlphaLight : kTrailAlphaDark};
        enc->setFragmentBytes(fade, sizeof(fade), NS::UInteger(0));
        enc->drawPrimitives(MTL::PrimitiveTypeTriangleStrip,
                            NS::UInteger(0), NS::UInteger(4));

        float rot[4] = {std::cos(angle_h), std::sin(angle_h),
                        std::cos(angle_v), std::sin(angle_v)};

        // 2. Reference frame, drawn under the flow so it never hides it
        if (showFrame && frameBuf) {
            enc->setRenderPipelineState(boxPSO);
            enc->setVertexBuffer(frameBuf, NS::UInteger(0), NS::UInteger(0));
            enc->setVertexBytes(rot, sizeof(rot), NS::UInteger(1));
            const int boxCfg[2] = {0, theme};
            enc->setFragmentBytes(boxCfg, sizeof(boxCfg), NS::UInteger(0));
            enc->drawPrimitives(MTL::PrimitiveTypeLine,
                                NS::UInteger(0), NS::UInteger(frameVertices));
        }

        // 3. Particles
        enc->setRenderPipelineState(pso);
        enc->setVertexBuffer(renderMetalBuf, renderBufOffset, NS::UInteger(0));
        enc->setVertexBytes(rot, sizeof(rot), NS::UInteger(1));
        const int cfg[2] = {colorMode, theme};
        enc->setFragmentBytes(cfg, sizeof(cfg), NS::UInteger(0));
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
        if (boxPSO)           boxPSO->release();
        if (frameBuf)         frameBuf->release();
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
    std::string projectorFilename;
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
                  << " [--no-vsync] [--fps] [--steps-per-frame=N]"
                     " [--projector FILE.nc]\n";
    };

    for (int i = 1; i < argc; i++) {
        const std::string_view arg = argv[i];
        constexpr std::string_view prefix = "--steps-per-frame=";
        constexpr std::string_view projectorPrefix = "--projector=";
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
        } else if (arg.starts_with(projectorPrefix)) {
            projectorFilename = arg.substr(projectorPrefix.size());
            if (projectorFilename.empty()) {
                usage();
                return 1;
            }
        } else if (arg == "--projector") {
            if (++i == argc || std::string_view(argv[i]).empty()) {
                usage();
                return 1;
            }
            projectorFilename = argv[i];
        } else {
            usage();
            return 1;
        }
    }

    std::optional<ProjectorInput> projectorInput;
    try {
        if (!projectorFilename.empty())
            projectorInput.emplace(load_projector(projectorFilename));
    } catch (const std::exception& error) {
        std::cerr << "cannot load projector: " << error.what() << "\n";
        return 1;
    }

    Demo demo(std::move(projectorInput));

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
                case SDLK_g:      demo.showFrame = !demo.showFrame; break;
                case SDLK_b:      demo.toggle_theme();            break;
                case SDLK_p:      demo.dump_eps();               break;
                case SDLK_f:      demo.apply_spectral_filter(); break;
                case SDLK_r:      demo.reset_particles();      break;
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
