#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"

#include <cstdlib>
#include <cstring>

static void test_cpu_weight_placement(ggml_backend_t npu) {
    auto cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu);
    ggml_backend_t backends[] = {npu, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    auto weights = ggml_init({ggml_tensor_overhead(), nullptr, true});
    auto w = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 128, 128);
    auto buffer = ggml_backend_alloc_ctx_tensors(weights, cpu);
    GGML_ASSERT(buffer);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto ctx = ggml_init({2 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 2);
    auto product = ggml_mul_mat(ctx, w, input);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, product);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched, product) == cpu);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(weights);
    ggml_backend_free(cpu);
}

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("OPENVINO0", nullptr);
    GGML_ASSERT(backend);
    const char * device = std::getenv("GGML_OPENVINO_DEVICE");
    if (device && std::strcmp(device, "NPU") == 0) {
        test_cpu_weight_placement(backend);
    }

    auto ctx = ggml_init({ggml_tensor_overhead() * 6 + 2 * ggml_graph_overhead(), nullptr, true});
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    auto scaled = ggml_scale(ctx, input, 2.0f);
    auto window = ggml_view_2d(ctx, scaled, 3, 2, scaled->nb[1], sizeof(float));
    auto consumer = ggml_sqr(ctx, window);
    ggml_set_name(input, "input");
    ggml_set_name(scaled, "scaled");
    ggml_set_name(window, "window");
    ggml_set_name(consumer, "consumer");
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, consumer);
    auto reshaped = ggml_reshape_2d(ctx, input, 4, 4);
    auto squares = ggml_sqr(ctx, reshaped);
    ggml_set_name(reshaped, "reshaped");
    ggml_set_name(squares, "squares");
    auto square_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(square_graph, squares);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);

    float values[16];
    for (int i = 0; i < 16; ++i) {
        values[i] = float(i + 1);
    }
    ggml_backend_tensor_set(input, values, 0, sizeof(values));
    ggml_backend_tensor_memset(scaled, 0, 0, sizeof(values));
    ggml_cgraph split = *graph;
    GGML_ASSERT(split.nodes[split.n_nodes - 1] == consumer);
    --split.n_nodes; // The next backend reads the window using GGML's row stride.
    GGML_ASSERT(ggml_backend_graph_compute(backend, &split) == GGML_STATUS_SUCCESS);
    for (int row = 0; row < 2; ++row) {
        float result[3];
        ggml_backend_tensor_get(window, result, row * window->nb[1], sizeof(result));
        for (int col = 0; col < 3; ++col) {
            GGML_ASSERT(result[col] == 2.0f * values[row * 8 + col + 1]);
        }
    }
    // The input to this fragment has the same storage as input, but different dimensions.
    ggml_cgraph tail = *square_graph;
    GGML_ASSERT(tail.n_nodes == 2 && tail.nodes[0] == reshaped);
    ++tail.nodes;
    --tail.n_nodes;
    GGML_ASSERT(ggml_backend_graph_compute(backend, &tail) == GGML_STATUS_SUCCESS);
    float result[16];
    ggml_backend_tensor_get(squares, result, 0, sizeof(result));
    for (int i = 0; i < 16; ++i) {
        GGML_ASSERT(result[i] == values[i] * values[i]);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
