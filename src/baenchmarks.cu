#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../include/cuda_utils.cuh"
#include "../include/sobel.cuh"

struct Result {
    std::string name;
    double ms = 0.0;
    double speedup = 0.0;
    double gbps = 0.0;
    double mpix = 0.0; // millions de pixels par seconde
    size_t bad = 0;
};

static size_t countMismatches(const std::vector<unsigned char>& a,
                              const std::vector<unsigned char>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += (a[i] != b[i]);
    return n;
}

// Un kernel est trop court (dizaines de µs) pour être mesuré seul : chaque exécution
// chronométrée enchaîne `inner` lancements. Le résultat est toujours affiché.
template <typename Launcher>
static Result evaluate(const char* name, Launcher&& launch, DeviceBuffer<unsigned char>& d_out,
                       const std::vector<unsigned char>& h_ref, size_t total_bytes,
                       double ms_cpu) {
    std::printf("  [en cours] %s ...\n", name);

    const int inner = 20;
    const float ms_batch = benchmarkMs([] {}, [&] { for (int i = 0; i < inner; ++i) launch(); },
                                       5, 20);
    const double ms = ms_batch / inner;

    // Validation : sentinelle 0xAA pour détecter tout pixel non écrit.
    CUDA_CHECK(cudaMemset(d_out.get(), 0xAA, total_bytes));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> h_gpu;
    d_out.download(h_gpu);

    Result r;
    r.name = name;
    r.ms = ms;
    r.speedup = ms_cpu / ms;
    r.gbps = (2.0 * total_bytes / 1e9) / (ms / 1000.0); // lu + écrit, modèle idéal
    r.mpix = (static_cast<double>(total_bytes) / 1e6) / (ms / 1000.0);
    r.bad = countMismatches(h_gpu, h_ref);

    std::printf("    -> %.5f ms | x%.1f vs CPU | %.1f GB/s effectifs | %.0f Mpix/s | écarts : %zu | %s\n",
                r.ms, r.speedup, r.gbps, r.mpix, r.bad, r.bad == 0 ? "OK" : "ÉCHEC");
    return r;
}

// Exécute tout le protocole pour une taille d'image donnée.
// Les buffers GPU sont libérés automatiquement en sortie de fonction (RAII).
static void runSize(int width, int height) {
    const size_t total = static_cast<size_t>(width) * height;
    const double mbytes = static_cast<double>(total) / (1024.0 * 1024.0);

    std::printf("\n=============================================================\n");
    std::printf("=== Sobel %dx%d (%.1f Mo) ===\n", width, height, mbytes);
    std::printf("=============================================================\n");

    // Image synthétique : texture sinusoïdale + bruit léger + carré blanc central.
    std::printf("Génération de l'image de test...\n");
    std::vector<unsigned char> h_in(total);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> noise(0, 15);
    const int sq0 = width / 4, sq1 = (3 * width) / 4;
    const int sy0 = height / 4, sy1 = (3 * height) / 4;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int v = static_cast<int>(128.0f + 90.0f * std::sin(0.05f * x) * std::cos(0.03f * y))
                    + noise(gen);
            if (x >= sq0 && x < sq1 && y >= sy0 && y < sy1) v = 255;
            h_in[static_cast<size_t>(y) * width + x] = static_cast<unsigned char>(v);
        }
    }

    // --- CPU : moyenne de 3 exécutions ---
    std::printf("[CPU] référence séquentielle (3 exécutions)...\n");
    std::vector<unsigned char> h_ref(total, 0);
    const int cpu_reps = 3;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < cpu_reps; ++i) sobelCPU(h_in, h_ref, width, height);
    const double ms_cpu =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / cpu_reps;
    std::printf("  -> CPU séquentiel : %.4f ms | %.0f Mpix/s\n\n", ms_cpu,
                (static_cast<double>(total) / 1e6) / (ms_cpu / 1000.0));

    // --- GPU ---
    DeviceBuffer<unsigned char> d_in(total), d_out(total);
    d_in.upload(h_in);

    std::printf("[GPU] mesures (5 warm-up + 20 mesures x 20 lancements par variante)\n");
    std::vector<Result> results;

    results.push_back(evaluate("GPU naïf (global memory)",
             [&] { launchSobelNaive(d_in.get(), d_out.get(), width, height); },
             d_out, h_ref, total, ms_cpu));

    results.push_back(evaluate("GPU tuilé (shared memory + halo)",
             [&] { launchSobelShared(d_in.get(), d_out.get(), width, height); },
             d_out, h_ref, total, ms_cpu));

    // --- Récapitulatif ---
    std::printf("\n=== RÉCAPITULATIF SOBEL (%dx%d) ===\n", width, height);
    std::printf("%-32s | %12s | %10s | %12s | %10s | %s\n",
                "Implémentation", "Temps (ms)", "Speedup", "GB/s eff.", "Écarts", "Statut");
    std::printf("---------------------------------+--------------+------------+--------------+------------+-------\n");
    std::printf("%-32s | %12.4f | %9.1fx | %12s | %10s | %s\n", "CPU séquentiel", ms_cpu, 1.0, "-",
                "(référence)", "-");
    for (const Result& r : results) {
        std::printf("%-32s | %12.5f | %9.1fx | %12.1f | %10zu | %s\n",
                    r.name.c_str(), r.ms, r.speedup, r.gbps, r.bad, r.bad == 0 ? "OK" : "ÉCHEC");
    }
    if (results.size() == 2) {
        const bool shared_faster = results[0].ms > results[1].ms;
        std::printf("\nShared vs naïf : %.2fx %s\n",
                    shared_faster ? results[0].ms / results[1].ms : results[1].ms / results[0].ms,
                    shared_faster ? "plus rapide" : "plus lent");
    }
}

int main() {
    // Sortie non bufferisée : tout ce qui est affiché reste visible même en cas de crash.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const auto t_program = std::chrono::steady_clock::now();

    std::printf("=== Audit de performance : filtre de Sobel ===\n");
    printDeviceInfo();

    // 2048x2048 (4 Mo)  : tient dans le L2 du GPU -> mesure dominée par le cache.
    // 8192x8192 (64 Mo) : dépasse le L2 -> mesure réellement limitée par la mémoire VRAM.
    runSize(2048, 2048);
    runSize(8192, 8192);

    const double total_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_program).count();
    std::printf("\nTemps total d'exécution du programme : %.2f s\n", total_s);
    std::printf("Benchmark terminé.\n");
    return 0;
}
