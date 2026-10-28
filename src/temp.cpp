// void MNNetwork::trainNetwork(const int iterations, Mnist::MnistData* image_data,
//                              bool* is_training) {
//   const int batch_size = m_batch_size;

//   for (int epoch = 0; epoch < iterations; ++epoch) {
//     for (int batch_start = 0;
//          batch_start < image_data->mnist_images_data.size();
//          batch_start += batch_size) {
//       if (!*is_training) return;

//       const int batch_end =
//           std::min(batch_start + batch_size,
//                    static_cast<int>(image_data->mnist_images_data.size()));

//       const int actual_batch_size = batch_end - batch_start;

//       // -----------------------------------------
//       // Get current network weights
//       // -----------------------------------------

//       std::vector<Mann::Matrix> weights;
//       std::vector<Mann::Matrix> biases;

//       {
//         std::lock_guard<std::mutex> lock(training_threads_mutex);

//         weights = MNN_Weights;
//         biases = MNN_Bias;
//       }

//       // -----------------------------------------
//       // Number of workers
//       // -----------------------------------------

//       const unsigned int worker_count =
//           std::min(4 u, static_cast<unsigned int>(actual_batch_size));

//       // -----------------------------------------
//       // Per-worker gradients
//       // -----------------------------------------

//       std::vector<NetworkGradients> worker_gradients;

//       for (unsigned int i = 0; i < worker_count; ++i) {
//         worker_gradients.emplace_back(weights, biases);
//       }

//       std::vector<int> worker_correct(worker_count, 0);

//       // -----------------------------------------
//       // Start workers
//       // -----------------------------------------

//       std::vector<std::thread> workers;

//       for (unsigned int worker = 0; worker < worker_count; ++worker) {
//         workers.emplace_back([&, worker]() {
//           int start = batch_start + (actual_batch_size * worker) / worker_count;

//           int end =
//               batch_start + (actual_batch_size * (worker + 1)) / worker_count;

//           NetworkGradients& gradients = worker_gradients[worker];

//           for (int i = start; i < end; ++i) {
//             if (!*is_training) return;

//             // Local network state
//             std::vector<Mann::Matrix> nodes = MNN_Nodes;

//             std::vector<Mann::Matrix> weighted_sum = biases;

//             Mann::Matrix target(MNN_Layers_size.back(), 1);

//             // -------------------------
//             // Input
//             // -------------------------

//             for (int j = 0; j < nodes[0].rows(); ++j) {
//               nodes[0][j] = image_data - > mnist_images_data[i][j];
//             }

//             // -------------------------
//             // Target
//             // -------------------------

//             for (int j = 0; j < target.rows(); ++j) {
//               target[j] = image_data - > mnist_labels_data[i][j];
//             }

//             // -------------------------
//             // Forward pass
//             // -------------------------

//             feedForward(nodes, weighted_sum, weights, biases);

//             // -------------------------
//             // Accuracy
//             // -------------------------

//             if (IsPredictionCorrect(nodes.back(), target)) {
//               worker_correct[worker]++;
//             }

//             // -------------------------
//             // Backprop
//             // -------------------------

//             NetworkGradients sample_gradient(weights, biases);

//             backPropagation(nodes, weighted_sum, weights, biases, target,
//                             sample_gradient);

//             // -------------------------
//             // ACCUMULATE
//             // -------------------------

//             for (size_t j = 0; j < gradients.weights.size(); ++j) {
//               gradients.weights[j] =
//                   gradients.weights[j] + sample_gradient.weights[j];

//               gradients.biases[j] =
//                   gradients.biases[j] + sample_gradient.biases[j];
//             }
//           }
//         });
//       }

//       // -----------------------------------------
//       // Wait for workers
//       // -----------------------------------------

//       for (auto& worker : workers) {
//         if (worker.joinable()) worker.join();
//       }

//       // -----------------------------------------
//       // Combine gradients
//       // -----------------------------------------

//       NetworkGradients final_gradient(weights, biases);

//       int correct_pred = 0;

//       for (unsigned int worker = 0; worker < worker_count; ++worker) {
//         correct_pred += worker_correct[worker];

//         for (size_t j = 0; j < final_gradient.weights.size(); ++j) {
//           final_gradient.weights[j] =
//               final_gradient.weights[j] + worker_gradients[worker].weights[j];

//           final_gradient.biases[j] =
//               final_gradient.biases[j] + worker_gradients[worker].biases[j];
//         }
//       }

//       // -----------------------------------------
//       // Average gradient
//       // -----------------------------------------

//       for (size_t j = 0; j < final_gradient.weights.size(); ++j) {
//         final_gradient.weights[j] =
//             final_gradient.weights[j] / static_cast<float>(actual_batch_size);

//         final_gradient.biases[j] =
//             final_gradient.biases[j] / static_cast<float>(actual_batch_size);
//       }

//       // -----------------------------------------
//       // ONE network update
//       // -----------------------------------------

//       {
//         std::lock_guard<std::mutex> lock(training_threads_mutex);

//         for (size_t j = 0; j < MNN_Weights.size(); ++j) {
//           MNN_Weights[j] =
//               MNN_Weights[j] - final_gradient.weights[j] * m_learning_rate;

//           MNN_Bias[j] =
//               MNN_Bias[j] - final_gradient.biases[j] * m_learning_rate;
//         }
//       }

//       // -----------------------------------------
//       // Batch statistics
//       // -----------------------------------------

//       m_batch_accuracy = static_cast<float>(correct_pred) /
//                          static_cast<float>(actual_batch_size) * 100.0 f;

//       m_batch_accuracy_history.push(m_batch_accuracy);
//     }

//     m_current_epoch++;
//   }
// }