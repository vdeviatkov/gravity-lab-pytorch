#include "gravity_lab/classic_environment.hpp"
#include "gravity_lab/classic_renderer.hpp"
#include "gravity_lab/dense_policy.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr int kWidth = 800;
constexpr int kHeight = 600;
constexpr std::array<std::string_view, 3> kGroups{"Easy", "Medium", "Pro"};
constexpr std::array<std::string_view, 4> kLeagues{"100cc", "175cc", "220cc", "325cc"};
constexpr std::array<int, 5> kFpsValues{25, 50, 100, 250, 0};
constexpr std::array<std::uint32_t, 5> kEpisodeValues{1, 5, 20, 100, 10'000};

struct Options {
    std::filesystem::path policy;
    std::uint64_t seed{2'000'007};
    std::uint32_t frame_skip{2};
    std::uint32_t max_steps{2'000};
    bool validate_only{false};
    bool catalog_only{false};
};

template <typename T>
T integer(std::string_view text, std::string_view option) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error("invalid value for " + std::string(option));
    }
    return value;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view {
            if (++i >= argc) throw std::runtime_error("missing value after " + std::string(arg));
            return argv[i];
        };
        if (arg == "--policy") options.policy = std::string(next());
        else if (arg == "--seed") options.seed = integer<std::uint64_t>(next(), arg);
        else if (arg == "--frame-skip") options.frame_skip = integer<std::uint32_t>(next(), arg);
        else if (arg == "--max-steps") options.max_steps = integer<std::uint32_t>(next(), arg);
        else if (arg == "--validate-only") options.validate_only = true;
        else if (arg == "--catalog-only") options.catalog_only = true;
        else if (arg == "--help") {
            std::cout << "Usage: gravity_lab_ai_arcade --policy FILE [options]\n"
                         "  --seed N --frame-skip N --max-steps N --validate-only --catalog-only\n"
                         "The graphical selector unlocks all groups, tracks, and leagues.\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + std::string(arg));
    }
    if (options.policy.empty()) throw std::runtime_error("--policy is required");
    return options;
}

void validate(const gravity_lab::DenseQPolicy& policy) {
    // A policy trained before the obstacle-ray sensor (or before acceleration) was added
    // declares a smaller observation_size; every prefix length in
    // [kBaseObservationSize, kObservationSize] is a compatible, unchanged prefix of the current
    // vector (see classic_environment.hpp), so all of them are accepted here and the observation
    // is truncated to the policy's own size when actually stepping.
    if (policy.environment_id() != "gravity-lab-classic-v1" ||
        policy.observation_size() < gravity_lab::classic::kBaseObservationSize ||
        policy.observation_size() > gravity_lab::classic::kObservationSize ||
        policy.action_count() != static_cast<std::size_t>(gravity_lab::classic::kActionCount)) {
        throw std::runtime_error("policy is incompatible with gravity-lab-classic-v1 ("
                                  + std::to_string(gravity_lab::classic::kBaseObservationSize)
                                  + " to " + std::to_string(gravity_lab::classic::kObservationSize)
                                  + " observations, 9 actions)");
    }
}

// The number of obstacle rays to compute so a policy's own observation region is populated with
// real values: derived from its declared observation_size (see classic_environment.hpp for the
// region layout), clamped to a valid ray count regardless of whether the policy uses any rays.
std::uint32_t obstacle_ray_count_for(const gravity_lab::DenseQPolicy& policy) {
    const auto base = gravity_lab::classic::kBaseObservationSize;
    const auto max_rays = gravity_lab::classic::kMaxObstacleRayCount;
    const std::size_t requested = policy.observation_size() > base ? policy.observation_size() - base : 0;
    return static_cast<std::uint32_t>(std::clamp<std::size_t>(requested, 1, max_rays));
}

using Catalog = std::array<std::vector<std::string>, 3>;

Catalog load_catalog(const Options& options) {
    Catalog catalog;
    for (std::uint32_t group = 0; group < catalog.size(); ++group) {
        gravity_lab::classic::Config base{group, 0, 0, options.frame_skip, options.max_steps, options.seed};
        std::uint32_t count = 0;
        {
            gravity_lab::classic::Environment environment(base);
            count = environment.track_count(group);
        }
        for (std::uint32_t track = 0; track < count; ++track) {
            base.track = track;
            gravity_lab::classic::Environment environment(base);
            catalog[group].push_back(environment.track_name());
        }
    }
    return catalog;
}

struct SdlDeleter {
    void operator()(SDL_Window* value) const { if (value) SDL_DestroyWindow(value); }
    void operator()(SDL_Renderer* value) const { if (value) SDL_DestroyRenderer(value); }
    void operator()(TTF_Font* value) const { if (value) TTF_CloseFont(value); }
    void operator()(SDL_Texture* value) const { if (value) SDL_DestroyTexture(value); }
};

using WindowPtr = std::unique_ptr<SDL_Window, SdlDeleter>;
using RendererPtr = std::unique_ptr<SDL_Renderer, SdlDeleter>;
using FontPtr = std::unique_ptr<TTF_Font, SdlDeleter>;
using TexturePtr = std::unique_ptr<SDL_Texture, SdlDeleter>;

void text(SDL_Renderer* renderer, TTF_Font* font, const std::string& value,
          int x, int y, SDL_Color color = {235, 240, 248, 255}) {
    SDL_Surface* surface = TTF_RenderUTF8_Blended(font, value.c_str(), color);
    if (!surface) throw std::runtime_error(TTF_GetError());
    std::unique_ptr<SDL_Surface, decltype(&SDL_FreeSurface)> owned(surface, SDL_FreeSurface);
    TexturePtr texture(SDL_CreateTextureFromSurface(renderer, surface));
    if (!texture) throw std::runtime_error(SDL_GetError());
    const SDL_Rect target{x, y, surface->w, surface->h};
    SDL_RenderCopy(renderer, texture.get(), nullptr, &target);
}

struct Selection {
    int row{0};
    int group{0};
    int track{0};
    int league{0};
    int fps_index{2};
    int episodes_index{1};
};

enum class MenuResult { Start, Quit };

void adjust(Selection& selection, int direction, const Catalog& catalog) {
    auto wrap = [direction](int value, int size) { return (value + direction + size) % size; };
    switch (selection.row) {
    case 0:
        selection.group = wrap(selection.group, 3);
        selection.track = std::min(selection.track,
                                   static_cast<int>(catalog[selection.group].size()) - 1);
        break;
    case 1: selection.track = wrap(selection.track, static_cast<int>(catalog[selection.group].size())); break;
    case 2: selection.league = wrap(selection.league, 4); break;
    case 3: selection.fps_index = wrap(selection.fps_index, static_cast<int>(kFpsValues.size())); break;
    case 4: selection.episodes_index = wrap(selection.episodes_index,
                                            static_cast<int>(kEpisodeValues.size())); break;
    default: break;
    }
}

MenuResult show_menu(Selection& selection, const Catalog& catalog) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) throw std::runtime_error(SDL_GetError());
    if (TTF_Init() != 0) { SDL_Quit(); throw std::runtime_error(TTF_GetError()); }
    struct Quitter { ~Quitter() { TTF_Quit(); SDL_Quit(); } } quitter;
    WindowPtr window(SDL_CreateWindow("Gravity Lab - AI Arcade", SDL_WINDOWPOS_CENTERED,
                                      SDL_WINDOWPOS_CENTERED, kWidth, kHeight, SDL_WINDOW_SHOWN));
    if (!window) throw std::runtime_error(SDL_GetError());
    RendererPtr renderer(SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_ACCELERATED));
    if (!renderer) renderer.reset(SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_SOFTWARE));
    if (!renderer) throw std::runtime_error(SDL_GetError());
    FontPtr title_font(TTF_OpenFont(GRAVITY_LAB_AI_FONT_PATH, 34));
    FontPtr font(TTF_OpenFont(GRAVITY_LAB_AI_FONT_PATH, 24));
    FontPtr small(TTF_OpenFont(GRAVITY_LAB_AI_FONT_PATH, 17));
    if (!title_font || !font || !small) throw std::runtime_error(TTF_GetError());

    while (true) {
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) return MenuResult::Quit;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) return MenuResult::Quit;
                if (event.key.keysym.sym == SDLK_UP) selection.row = (selection.row + 5) % 6;
                if (event.key.keysym.sym == SDLK_DOWN) selection.row = (selection.row + 1) % 6;
                if (event.key.keysym.sym == SDLK_LEFT) adjust(selection, -1, catalog);
                if (event.key.keysym.sym == SDLK_RIGHT) adjust(selection, 1, catalog);
                if ((event.key.keysym.sym == SDLK_RETURN || event.key.keysym.sym == SDLK_SPACE) &&
                    selection.row == 5) return MenuResult::Start;
            }
            if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
                const int row = (event.button.y - 165) / 55;
                if (row >= 0 && row < 6) {
                    selection.row = row;
                    if (row == 5) return MenuResult::Start;
                    adjust(selection, event.button.x < kWidth / 2 ? -1 : 1, catalog);
                }
            }
        }

        SDL_SetRenderDrawColor(renderer.get(), 18, 24, 35, 255);
        SDL_RenderClear(renderer.get());
        text(renderer.get(), title_font.get(), "Gravity Lab: AI Arcade", 190, 40,
             {100, 210, 255, 255});
        text(renderer.get(), small.get(), "Everything unlocked - the trained policy drives", 220, 95);
        const std::array<std::string, 6> values{
            std::string(kGroups[selection.group]),
            catalog[selection.group][selection.track] + "  (#" + std::to_string(selection.track) + ")",
            std::string(kLeagues[selection.league]),
            kFpsValues[selection.fps_index] == 0 ? "Unlimited" : std::to_string(kFpsValues[selection.fps_index]),
            kEpisodeValues[selection.episodes_index] == 10'000 ? "Until Esc" :
                std::to_string(kEpisodeValues[selection.episodes_index]),
            "START AI",
        };
        constexpr std::array<std::string_view, 6> labels{
            "Level", "Track", "League", "Playback FPS", "Episodes", ""
        };
        for (int row = 0; row < 6; ++row) {
            const int y = 165 + row * 55;
            if (row == selection.row) {
                SDL_SetRenderDrawColor(renderer.get(), 42, 78, 105, 255);
                const SDL_Rect highlight{105, y - 7, 590, 45};
                SDL_RenderFillRect(renderer.get(), &highlight);
            }
            if (row < 5) {
                text(renderer.get(), font.get(), std::string(labels[row]), 130, y);
                text(renderer.get(), font.get(), "<  " + values[row] + "  >", 390, y,
                     {255, 220, 105, 255});
            } else {
                text(renderer.get(), font.get(), values[row], 325, y, {115, 245, 145, 255});
            }
        }
        text(renderer.get(), small.get(), "Arrow keys: choose   Enter: start   Esc: quit", 235, 520,
             {170, 180, 195, 255});
        text(renderer.get(), small.get(), "During playback, Esc returns to this selector", 225, 548,
             {170, 180, 195, 255});
        SDL_RenderPresent(renderer.get());
        SDL_Delay(16);
    }
}

class FramePacer {
public:
    explicit FramePacer(int fps) : enabled_(fps > 0), next_(std::chrono::steady_clock::now()) {
        if (enabled_) interval_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / fps));
    }
    void wait() {
        if (!enabled_) return;
        next_ += interval_;
        std::this_thread::sleep_until(next_);
    }
private:
    bool enabled_{};
    std::chrono::steady_clock::time_point next_;
    std::chrono::steady_clock::duration interval_{};
};

void play(const Options& options, const Selection& selection,
          const gravity_lab::DenseQPolicy& policy) {
    gravity_lab::classic::Config config{
        static_cast<std::uint32_t>(selection.group), static_cast<std::uint32_t>(selection.track),
        static_cast<std::uint32_t>(selection.league), options.frame_skip, options.max_steps, options.seed,
        obstacle_ray_count_for(policy)};
    gravity_lab::classic::Environment environment(config);
    gravity_lab::classic::Renderer renderer(
        environment, "Gravity Lab AI - " + environment.track_name() + " - " +
                     std::string(kLeagues[selection.league]));
    FramePacer pacer(kFpsValues[selection.fps_index]);
    const auto episodes = kEpisodeValues[selection.episodes_index];
    for (std::uint32_t episode = 0; episode < episodes && renderer.open(); ++episode) {
        auto observation = environment.reset(options.seed + episode);
        gravity_lab::classic::StepResult result;
        double reward = 0.0;
        std::uint64_t elapsed = 0;
        renderer.show_message(environment.track_name(), 700);
        while (!environment.done() && renderer.open()) {
            const auto action = policy.action(
                std::span<const double>(observation.data(), policy.observation_size()));
            result = environment.step(static_cast<gravity_lab::classic::Action>(action));
            observation = result.observation;
            reward += result.reward;
            elapsed += 20ULL * options.frame_skip;
            renderer.render_frame(elapsed);
            pacer.wait();
        }
        if (!renderer.open()) break;
        renderer.show_message(result.finished ? "Finished" : result.crashed ? "Crashed" : "Time limit", 600);
        std::cout << "episode=" << episode << " track=\"" << environment.track_name()
                  << "\" reward=" << reward << " progress=" << observation[0]
                  << " finished=" << result.finished << " crashed=" << result.crashed
                  << " truncated=" << result.truncated << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const auto policy = gravity_lab::DenseQPolicy::load(options.policy);
        validate(policy);
        if (options.validate_only) {
            std::cout << "valid AI arcade policy observations=" << policy.observation_size()
                      << " actions=" << policy.action_count() << '\n';
            return 0;
        }
        const auto catalog = load_catalog(options);
        if (options.catalog_only) {
            for (std::size_t group = 0; group < catalog.size(); ++group) {
                std::cout << "group=" << group << " name=" << kGroups[group]
                          << " tracks=" << catalog[group].size() << '\n';
                for (std::size_t track = 0; track < catalog[group].size(); ++track) {
                    std::cout << "  track=" << track << " name=\"" << catalog[group][track] << "\"\n";
                }
            }
            std::cout << "leagues=4 (100cc,175cc,220cc,325cc)\n";
            return 0;
        }
        Selection selection;
        while (show_menu(selection, catalog) == MenuResult::Start) play(options, selection, policy);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
