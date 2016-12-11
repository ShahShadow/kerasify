/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "keras_model.h"

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <utility>

namespace kerasify {

bool ReadUnsignedInt(std::ifstream* file, unsigned int* i) {
  KASSERT(file, "Invalid file stream");
  KASSERT(i, "Invalid pointer");

  file->read((char*)i, sizeof(unsigned int));
  KASSERT(file->gcount() == sizeof(unsigned int), "Expected unsigned int");

  return true;
}

bool ReadFloat(std::ifstream* file, float* f) {
  KASSERT(file, "Invalid file stream");
  KASSERT(f, "Invalid pointer");

  file->read((char*)f, sizeof(float));
  KASSERT(file->gcount() == sizeof(float), "Expected float");

  return true;
}

bool ReadFloats(std::ifstream* file, float* f, size_t n) {
  KASSERT(file, "Invalid file stream");
  KASSERT(f, "Invalid pointer");

  file->read((char*)f, sizeof(float) * n);
  KASSERT(((unsigned int)file->gcount()) == sizeof(float) * n,
          "Expected floats");

  return true;
}

bool ReadString(std::ifstream* file, std::string* str) {
  KASSERT(file, "Invalid file stream");
  KASSERT(str, "Invalid pointer");

  unsigned int n;
  KASSERT(ReadUnsignedInt(file, &n), "Expected string size");

  char buffer[n];
  file->read((char*)&buffer, sizeof(char) * n);
  *str = std::string((char*)&buffer);
  KASSERT(((unsigned int)file->gcount()) == sizeof(char) * n, "Expected chars");

  return true;
}

bool ReadStrings(std::ifstream* file, std::vector<std::string>* strs) {
  KASSERT(file, "Invalid file stream");
  KASSERT(strs, "Invalid pointer");

  unsigned int n;
  KASSERT(ReadUnsignedInt(file, &n), "Expected string list count");

  strs->clear();
  strs->resize(n);
  for (unsigned int i = 0; i < n; i++) {
    KASSERT(ReadString(file, &((*strs)[i])), "Expected string in list");
  }

  return true;
}

bool KerasLayerInput::LoadLayer(std::ifstream* file) {
  KASSERT(file, "Invalid file stream");

  return true;
}

bool KerasLayerInput::Apply(const std::vector<Tensor*>& in_list, Tensor* out) {
  KASSERT(in_list.size() == 1, "Invalid input");
  KASSERT(out, "Invalid output");

  *out = *in_list[0];

  return true;
}

bool KerasLayerActivation::LoadLayer(std::ifstream* file) {
  KASSERT(file, "Invalid file stream");

  unsigned int activation = 0;
  KASSERT(ReadUnsignedInt(file, &activation), "Failed to read activation type");

  switch (activation) {
    case kLinear:
      activation_type_ = kLinear;
      break;
    case kRelu:
      activation_type_ = kRelu;
      break;
    case kSoftPlus:
      activation_type_ = kSoftPlus;
      break;
    default:
      KASSERT(false, "Unsupported activation type %d", activation);
  }

  return true;
}

bool KerasLayerActivation::Apply(const std::vector<Tensor*>& in_list,
                                 Tensor* out) {
  KASSERT(in_list.size() == 1, "Invalid input");
  KASSERT(out, "Invalid output");

  *out = *in_list[0];

  switch (activation_type_) {
    case kLinear:
      break;
    case kRelu:
      for (size_t i = 0; i < out->data_.size(); i++) {
        if (out->data_[i] < 0.0) {
          out->data_[i] = 0.0;
        }
      }
      break;
    case kSoftPlus:
      for (size_t i = 0; i < out->data_.size(); i++) {
        out->data_[i] = std::log(1.0 + std::exp(out->data_[i]));
      }
      break;
    default:
      break;
  }

  return true;
}

bool KerasLayerDense::LoadLayer(std::ifstream* file) {
  KASSERT(file, "Invalid file stream");

  unsigned int weights_rows = 0;
  KASSERT(ReadUnsignedInt(file, &weights_rows), "Expected weight rows");
  KASSERT(weights_rows > 0, "Invalid weights # rows");

  unsigned int weights_cols = 0;
  KASSERT(ReadUnsignedInt(file, &weights_cols), "Expected weight cols");
  KASSERT(weights_cols > 0, "Invalid weights shape");

  unsigned int biases_shape = 0;
  KASSERT(ReadUnsignedInt(file, &biases_shape), "Expected biases shape");
  KASSERT(biases_shape > 0, "Invalid biases shape");

  weights_.Resize(weights_rows, weights_cols);
  KASSERT(ReadFloats(file, weights_.data_.data(), weights_rows * weights_cols),
          "Expected weights");

  biases_.Resize(biases_shape);
  KASSERT(ReadFloats(file, biases_.data_.data(), biases_shape),
          "Expected biases");

  KASSERT(activation_.LoadLayer(file), "Failed to load activation");

  return true;
}

bool KerasLayerDense::Apply(const std::vector<Tensor*>& in_list, Tensor* out) {
  KASSERT(in_list.size() == 1, "Invalid input");
  KASSERT(out, "Invalid output");

  Tensor* in = in_list[0];
  KASSERT(in->dims_.size() <= 2, "Invalid input dimensions");

  if (in->dims_.size() == 2) {
    KASSERT(in->dims_[1] == weights_.dims_[0], "Dimension mismatch %d %d",
            in->dims_[1], weights_.dims_[0]);
  }

  Tensor tmp(weights_.dims_[1]);

  for (int i = 0; i < weights_.dims_[0]; i++) {
    for (int j = 0; j < weights_.dims_[1]; j++) {
      tmp(j) += (*in)(i)*weights_(i, j);
    }
  }

  for (int i = 0; i < biases_.dims_[0]; i++) {
    tmp(i) += biases_(i);
  }

  KASSERT(activation_.Apply({&tmp}, out), "Failed to apply activation");

  return true;
}

bool KerasLayerConvolution2d::LoadLayer(std::ifstream* file) {
  KASSERT(file, "Invalid file stream");

  unsigned int weights_i = 0;
  KASSERT(ReadUnsignedInt(file, &weights_i), "Expected weights_i");
  KASSERT(weights_i > 0, "Invalid weights # i");

  unsigned int weights_j = 0;
  KASSERT(ReadUnsignedInt(file, &weights_j), "Expected weights_j");
  KASSERT(weights_j > 0, "Invalid weights # j");

  unsigned int weights_k = 0;
  KASSERT(ReadUnsignedInt(file, &weights_k), "Expected weights_k");
  KASSERT(weights_k > 0, "Invalid weights # k");

  unsigned int weights_l = 0;
  KASSERT(ReadUnsignedInt(file, &weights_l), "Expected weights_l");
  KASSERT(weights_l > 0, "Invalid weights # l");

  unsigned int biases_shape = 0;
  KASSERT(ReadUnsignedInt(file, &biases_shape), "Expected biases shape");
  KASSERT(biases_shape > 0, "Invalid biases shape");

  weights_.Resize(weights_i, weights_j, weights_k, weights_l);
  KASSERT(ReadFloats(file, weights_.data_.data(),
                     weights_i * weights_j * weights_k * weights_l),
          "Expected weights");

  biases_.Resize(biases_shape);
  KASSERT(ReadFloats(file, biases_.data_.data(), biases_shape),
          "Expected biases");

  KASSERT(activation_.LoadLayer(file), "Failed to load activation");

  return true;
}

bool KerasLayerConvolution2d::Apply(const std::vector<Tensor*>& in_list,
                                    Tensor* out) {
  KASSERT(in_list.size() == 1, "Invalid input");
  KASSERT(out, "Invalid output");

  Tensor* in = in_list[0];
  KASSERT(in->dims_[0] == weights_.dims_[1],
          "Input 'depth' doesn't match kernel 'depth'");

  int st_nj = (weights_.dims_[2] - 1) / 2;
  int st_pj = (weights_.dims_[2]) / 2;
  int st_nk = (weights_.dims_[3] - 1) / 2;
  int st_pk = (weights_.dims_[3]) / 2;

  Tensor tmp(weights_.dims_[0], in->dims_[1] - st_nj - st_pj,
             in->dims_[2] - st_nk - st_pk);

  // Iterate over each kernel.
  for (int i = 0; i < weights_.dims_[0]; i++) {
    // Iterate over each 'depth'.
    for (int j = 0; j < weights_.dims_[1]; j++) {
      // 2D convolution in x and y (k and l in Tensor dimensions).
      for (int tj = st_nj; tj < in->dims_[1] - st_pj; tj++) {
        for (int tk = st_nk; tk < in->dims_[2] - st_pk; tk++) {
          // Iterate over kernel.
          for (int k = 0; k < weights_.dims_[2]; k++) {
            for (int l = 0; l < weights_.dims_[3]; l++) {
              const float& weight = weights_(i, j, k, l);
              const float& value = (*in)(j, tj - st_nj + k, tk - st_nk + l);

              tmp(i, tj - st_nj, tk - st_nk) += weight * value;
            }
          }
        }
      }
    }

    // Apply kernel bias to all points in output.
    for (int j = 0; j < tmp.dims_[1]; j++) {
      for (int k = 0; k < tmp.dims_[2]; k++) {
        tmp(i, j, k) += biases_(i);
      }
    }
  }

  KASSERT(activation_.Apply({&tmp}, out), "Failed to apply activation");

  return true;
}

bool KerasLayerFlatten::LoadLayer(std::ifstream* file) {
  KASSERT(file, "Invalid file stream");
  return true;
}

bool KerasLayerFlatten::Apply(const std::vector<Tensor*>& in_list,
                              Tensor* out) {
  KASSERT(in_list.size() == 1, "Invalid input");
  KASSERT(out, "Invalid output");

  *out = *in_list[0];
  out->Flatten();

  return true;
}

bool KerasLayerElu::LoadLayer(std::ifstream* file) {
  KASSERT(file, "Invalid file stream");

  KASSERT(ReadFloat(file, &alpha_), "Failed to read alpha");

  return true;
}

bool KerasLayerElu::Apply(const std::vector<Tensor*>& in_list, Tensor* out) {
  KASSERT(in_list.size() == 1, "Invalid input");
  KASSERT(out, "Invalid output");

  *out = *in_list[0];

  for (size_t i = 0; i < out->data_.size(); i++) {
    if (out->data_[i] < 0.0) {
      out->data_[i] = alpha_ * (exp(out->data_[i]) - 1.0);
    }
  }

  return true;
}

bool KerasLayerMaxPooling2d::LoadLayer(std::ifstream* file) {
  KASSERT(file, "Invalid file stream");

  KASSERT(ReadUnsignedInt(file, &pool_size_j_), "Expected pool size j");
  KASSERT(ReadUnsignedInt(file, &pool_size_k_), "Expected pool size k");

  return true;
}

bool KerasLayerMaxPooling2d::Apply(const std::vector<Tensor*>& in_list,
                                   Tensor* out) {
  KASSERT(in_list.size() == 1, "Invalid input");
  KASSERT(out, "Invalid output");

  Tensor* in = in_list[0];
  KASSERT(in->dims_.size() == 3, "Input must have 3 dimensions");

  Tensor tmp(in->dims_[0], in->dims_[1] / pool_size_j_,
             in->dims_[2] / pool_size_k_);

  for (int i = 0; i < tmp.dims_[0]; i++) {
    for (int j = 0; j < tmp.dims_[1]; j++) {
      const int tj = j * pool_size_j_;

      for (int k = 0; k < tmp.dims_[2]; k++) {
        const int tk = k * pool_size_k_;

        // Find maximum value over patch starting at tj, tk.
        float max_val = -std::numeric_limits<float>::infinity();

        for (unsigned int pj = 0; pj < pool_size_j_; pj++) {
          for (unsigned int pk = 0; pk < pool_size_k_; pk++) {
            const float& pool_val = (*in)(i, tj + pj, tk + pk);
            if (pool_val > max_val) {
              max_val = pool_val;
            }
          }
        }

        tmp(i, j, k) = max_val;
      }
    }
  }

  *out = tmp;

  return true;
}

bool KerasModel::LoadModel(const std::string& filename) {
  std::ifstream file(filename.c_str(), std::ios::binary);
  KASSERT(file.is_open(), "Unable to open file %s", filename.c_str());

  unsigned int num_layers = 0;
  KASSERT(ReadUnsignedInt(&file, &num_layers), "Expected number of layers");

  KASSERT(ReadStrings(&file, &input_layer_names_),
          "Expected input layer names");
  for (const std::string& layer_name : input_layer_names_) {
    printf("input: %s\n", layer_name.c_str());
  }
  KASSERT(!input_layer_names_.empty(),
          "Expected at least one output layer name.")
  KASSERT(ReadStrings(&file, &output_layer_names_),
          "Expected output layer names");
  KASSERT(!output_layer_names_.empty(),
          "Expected at least one output layer name.");
  for (const std::string& layer_name : output_layer_names_) {
    printf("output: %s\n", layer_name.c_str());
  }

  for (unsigned int i = 0; i < num_layers; i++) {
    std::string layer_name;
    KASSERT(ReadString(&file, &layer_name), "Expected layer name");

    printf("Layer name: %s\n", layer_name.c_str());

    std::vector<std::string> inbound_layer_names;
    KASSERT(ReadStrings(&file, &inbound_layer_names),
            "Expected inbound layer names");
    for (const std::string& inbound_layer_name : inbound_layer_names) {
      printf("Inbound layer name: %s\n", inbound_layer_name.c_str());
    }

    unsigned int layer_type = 0;
    KASSERT(ReadUnsignedInt(&file, &layer_type), "Expected layer type");

    KerasLayer* layer = nullptr;

    switch (layer_type) {
      case kDense:
        layer = new KerasLayerDense(layer_name, inbound_layer_names);
        break;
      case kConvolution2d:
        layer = new KerasLayerConvolution2d(layer_name, inbound_layer_names);
        break;
      case kFlatten:
        layer = new KerasLayerFlatten(layer_name, inbound_layer_names);
        break;
      case kElu:
        layer = new KerasLayerElu(layer_name, inbound_layer_names);
        break;
      case kActivation:
        layer = new KerasLayerActivation(layer_name, inbound_layer_names);
        break;
      case kMaxPooling2D:
        layer = new KerasLayerMaxPooling2d(layer_name, inbound_layer_names);
        break;
      case kInput:
        layer = new KerasLayerInput(layer_name, inbound_layer_names);
        break;
      default:
        break;
    }

    KASSERT(layer, "Unknown layer type %d", layer_type);

    KASSERT(layer->LoadLayer(&file), "Failed to load layer %d", i);
    layers_.push_back(layer);
  }

  return true;
}

bool KerasModel::Apply(Tensor* in, Tensor* out) {
  KASSERT(output_layer_names_.size() == 1,
          "Only single output models supported.");
  const std::string& output_layer_name = output_layer_names_[0];

  KASSERT(input_layer_names_.size() == 1,
          "Only single input models supported.");
  const std::string& input_layer_name = input_layer_names_[0];

  TensorMap out_map = {{output_layer_name, out}};
  return Apply({{input_layer_name, in}}, &out_map);
}

class KerasNode {
 public:
  explicit KerasNode(KerasLayer* layer) : layer_(layer) {}
  Tensor* Compute() {
    if (computed_) {
      return result_;
    }

    std::vector<Tensor*> in_list;
    for (const std::shared_ptr<KerasNode>& node : inbound_nodes_) {
      in_list.push_back(node->Compute());
    }

    KASSERT(layer_->Apply(in_list, &result_), "Failed to apply layer %s",
            layer_->name());
    computed_ = true;
  }

  void AddInboundNodes(
      const std::unordered_map<std::string, KerasLayer*>& layer_map,
      std::unordered_map<std::string, std::shared_ptr<KerasNode>>* node_map) {
    for (const std::string& layer_name : layer_->inbound_layer_names()) {
      if (node_map->find(layer_name) == node_map.end()) {
        (*node_map)[layer_name] =
            std::make_shared<KerasNode>(layer_map[layer_name]);
      }

      inbound_nodes_.push_back(node_map[layer_name]);
    }
  }

  void SetResult(const Tensor& in) {
    computed_ = true;
    result_ = in;
  }

  const std::string& name() const { return layer_->name(); }

 private:
  KerasLayer* layer_;
  std::vector<std::shared_ptr<KerasNode>> inbound_nodes_;

  Tensor result_;
  bool computed_ = false;
};

bool KerasModel::Apply(std::unordered_map<std::string, Tensor*>& in_map,
                       std::unordered_map<std::string, Tensor*>* out_map) {
  KASSERT(!in_map.empty(), "No inputs provided");
  KASSERT(out_map, "Invalid output map");
  KASSERT(!out_map->empty(), "No outputs requested");

  // Build layer map.
  std::unordered_map<std::string, KerasLayer*> layer_map;
  for (KerasLayer* layer : layers_) {
    layer_map[layer->name()] = layer;
  }

  // Build node map.
  std::unordered_map<std::string, std::shared_ptr<KerasNode>> node_map;
  std::vector<std::shared_ptr<KerasNode>> out_nodes;
  for (auto out_map_iter : out_map) {
    const std::string& layer_name = out_map_iter->first;

    // One out node may be dependent upon another.
    if (node_map.find(layer_name) == node_map.end()) {
      node_map[layer_name] = std::make_shared<KerasNode>(layer_map[layer_name]);
      node_map[layer_name].AddInboundNodes(layer_map, &node_map);
    }

    out_nodes.push_back(node_map[layer_name]);
  }

  // Set input on input nodes in graph.
  for (auto in_map_iter : in_map) {
    const std::string& layer_name = in_map_iter->first;
    Tensor* in = in_map_iter->second;
    node_map[layer_name]->SetResult(in);
  }

  // Compute output nodes.
  for (const std::shared_ptr<KerasNode>& out_node : out_nodes) {
    *(out_map[out_node.name()]) = out_node->Compute();
  }

  return true;
}

}  // namespace kerasify
