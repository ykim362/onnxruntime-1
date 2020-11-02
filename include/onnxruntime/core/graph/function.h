// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {
class Graph;
class Node;
}  // namespace onnxruntime

namespace onnxruntime {

/** 
@class Function 
Class representing a Function.
*/
class Function {
 public:
  virtual ~Function() = default;

  /** Gets the OpSchema for the Function. */
  virtual const ONNX_NAMESPACE::OpSchema& OpSchema() const = 0;

  /** Gets the Graph instance for the Function body subgraph. */
  virtual const onnxruntime::Graph& Body() const = 0;

  /** Gets the IndexedSubGraph for the Function. */
  virtual const IndexedSubGraph& GetIndexedSubGraph() const = 0;
};

/** 
Create a new Function instance.
@param graph The graph containing the Function.
@param nodes_to_fuse the IndexedSubGraph to use for the Function.
@param create_new_model If true will create a new Model in the Function body. If false a lightweight Function 
                        implementation will be returned that has an OpSchema but no Body or IndexedSubGraph.
*/
std::unique_ptr<Function> MakeFunction(const onnxruntime::Graph& graph,
                                       std::unique_ptr<IndexedSubGraph> nodes_to_fuse,
                                       const logging::Logger& logger);
}  // namespace onnxruntime
