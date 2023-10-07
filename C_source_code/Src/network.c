/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Fri Sep 30 23:48:49 2022
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"




#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "903836ba8df9464c2cfac3294cce8cea"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Fri Sep 30 23:48:49 2022"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)




/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  onnxMaxPool_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1250, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 312, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  input_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 312, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  input_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 515, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  input_15_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 165, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  input_23_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  onnxReshape_39_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  node_42_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  input_3_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  input_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  input_7_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 30, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  input_7_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  input_15_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  input_15_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  input_23_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  input_23_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  onnxReshape_39_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  onnxReshape_39_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  fc1_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  fc1_1_weight_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  onnxMaxPool_0_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1250), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &onnxMaxPool_0_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 312), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &input_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  input_3_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 312), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &input_3_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  input_7_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 103), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &input_7_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  input_15_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 33), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &input_15_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  input_23_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 10), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &input_23_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  onnxReshape_39_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 10), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &onnxReshape_39_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  onnxReshape_39_output0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &onnxReshape_39_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  node_42_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &node_42_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  input_3_scale, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &input_3_scale_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  input_3_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &input_3_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  input_7_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 6, 5), AI_STRIDE_INIT(4, 4, 4, 4, 24),
  1, &input_7_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  input_7_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &input_7_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  input_15_weights, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 5, 1, 6, 5), AI_STRIDE_INIT(4, 4, 20, 20, 120),
  1, &input_15_weights_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  input_15_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &input_15_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  input_23_weights, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 5, 1, 6, 5), AI_STRIDE_INIT(4, 4, 20, 20, 120),
  1, &input_23_weights_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  input_23_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &input_23_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  onnxReshape_39_scale, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &onnxReshape_39_scale_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  onnxReshape_39_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &onnxReshape_39_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  fc1_1_bias, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &fc1_1_bias_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  fc1_1_weight, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 2), AI_STRIDE_INIT(4, 4, 200, 400, 400),
  1, &fc1_1_weight_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  node_42_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &onnxReshape_39_output0, &fc1_1_weight, &fc1_1_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_42_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  node_42_layer, 12,
  GEMM_TYPE, 0x0, NULL,
  gemm, forward_gemm,
  &node_42_chain,
  NULL, &node_42_layer, AI_STATIC, 
  .alpha = 1.0, 
  .beta = 1.0, 
  .tA = 0, 
  .tB = 1, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  onnxReshape_39_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &onnxReshape_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &onnxReshape_39_scale, &onnxReshape_39_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  onnxReshape_39_layer, 10,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &onnxReshape_39_chain,
  NULL, &node_42_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_23_weights, &input_23_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_23_layer, 9,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_23_chain,
  NULL, &onnxReshape_39_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 3), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_15_weights, &input_15_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_15_layer, 6,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_15_chain,
  NULL, &input_23_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 3), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_7_weights, &input_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_7_layer, 3,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_7_chain,
  NULL, &input_15_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 3), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &input_3_scale, &input_3_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_3_layer, 1,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &input_3_chain,
  NULL, &input_7_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &onnxMaxPool_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_layer, 0,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &input_chain,
  NULL, &input_3_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 5), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 4), 
  .count_include_pad = 0, 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1836, 1, 1),
    1836, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2156, 1, 1),
    2156, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &onnxMaxPool_0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &node_42_output),
  &input_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1836, 1, 1),
      1836, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2156, 1, 1),
      2156, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &onnxMaxPool_0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &node_42_output),
  &input_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  ai_ptr activations_map[1] = AI_C_ARRAY_INIT;

  if (ai_platform_get_activations_map(activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    input_output_array.data = AI_PTR(activations_map[0] + 908);
    input_output_array.data_start = AI_PTR(activations_map[0] + 908);
    input_3_output_array.data = AI_PTR(activations_map[0] + 908);
    input_3_output_array.data_start = AI_PTR(activations_map[0] + 908);
    input_7_output_array.data = AI_PTR(activations_map[0] + 40);
    input_7_output_array.data_start = AI_PTR(activations_map[0] + 40);
    input_15_output_array.data = AI_PTR(activations_map[0] + 0);
    input_15_output_array.data_start = AI_PTR(activations_map[0] + 0);
    input_23_output_array.data = AI_PTR(activations_map[0] + 660);
    input_23_output_array.data_start = AI_PTR(activations_map[0] + 660);
    onnxReshape_39_output_array.data = AI_PTR(activations_map[0] + 0);
    onnxReshape_39_output_array.data_start = AI_PTR(activations_map[0] + 0);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  ai_ptr weights_map[1] = AI_C_ARRAY_INIT;

  if (ai_platform_get_weights_map(weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    input_3_scale_array.format |= AI_FMT_FLAG_CONST;
    input_3_scale_array.data = AI_PTR(weights_map[0] + 0);
    input_3_scale_array.data_start = AI_PTR(weights_map[0] + 0);
    input_3_bias_array.format |= AI_FMT_FLAG_CONST;
    input_3_bias_array.data = AI_PTR(weights_map[0] + 4);
    input_3_bias_array.data_start = AI_PTR(weights_map[0] + 4);
    input_7_weights_array.format |= AI_FMT_FLAG_CONST;
    input_7_weights_array.data = AI_PTR(weights_map[0] + 8);
    input_7_weights_array.data_start = AI_PTR(weights_map[0] + 8);
    input_7_bias_array.format |= AI_FMT_FLAG_CONST;
    input_7_bias_array.data = AI_PTR(weights_map[0] + 128);
    input_7_bias_array.data_start = AI_PTR(weights_map[0] + 128);
    input_15_weights_array.format |= AI_FMT_FLAG_CONST;
    input_15_weights_array.data = AI_PTR(weights_map[0] + 148);
    input_15_weights_array.data_start = AI_PTR(weights_map[0] + 148);
    input_15_bias_array.format |= AI_FMT_FLAG_CONST;
    input_15_bias_array.data = AI_PTR(weights_map[0] + 748);
    input_15_bias_array.data_start = AI_PTR(weights_map[0] + 748);
    input_23_weights_array.format |= AI_FMT_FLAG_CONST;
    input_23_weights_array.data = AI_PTR(weights_map[0] + 768);
    input_23_weights_array.data_start = AI_PTR(weights_map[0] + 768);
    input_23_bias_array.format |= AI_FMT_FLAG_CONST;
    input_23_bias_array.data = AI_PTR(weights_map[0] + 1368);
    input_23_bias_array.data_start = AI_PTR(weights_map[0] + 1368);
    onnxReshape_39_scale_array.format |= AI_FMT_FLAG_CONST;
    onnxReshape_39_scale_array.data = AI_PTR(weights_map[0] + 1388);
    onnxReshape_39_scale_array.data_start = AI_PTR(weights_map[0] + 1388);
    onnxReshape_39_bias_array.format |= AI_FMT_FLAG_CONST;
    onnxReshape_39_bias_array.data = AI_PTR(weights_map[0] + 1408);
    onnxReshape_39_bias_array.data_start = AI_PTR(weights_map[0] + 1408);
    fc1_1_bias_array.format |= AI_FMT_FLAG_CONST;
    fc1_1_bias_array.data = AI_PTR(weights_map[0] + 1428);
    fc1_1_bias_array.data_start = AI_PTR(weights_map[0] + 1428);
    fc1_1_weight_array.format |= AI_FMT_FLAG_CONST;
    fc1_1_weight_array.data = AI_PTR(weights_map[0] + 1436);
    fc1_1_weight_array.data_start = AI_PTR(weights_map[0] + 1436);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 12671,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 12671,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_network_data_params_get(&params) != true) {
        err = ai_network_get_error(*network);
        return err;
    }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_network_init(*network, &params) != true) {
        err = ai_network_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

