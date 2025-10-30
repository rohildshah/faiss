/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/** In this file are the implementations of extra metrics beyond L2
 *  and inner product */

#include <faiss/MetricType.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <cmath>
#include <type_traits>

namespace faiss {

template <MetricType mt>
struct VectorDistance {
    size_t d;
    float metric_arg;
    static constexpr bool is_similarity = is_similarity_metric(mt);

    inline float operator()(const float* x, const float* y) const;

    // heap template to use for this type of metric
    using C = typename std::conditional<
            is_similarity_metric(mt),
            CMin<float, int64_t>,
            CMax<float, int64_t>>::type;
};

template <>
inline float VectorDistance<METRIC_L2>::operator()(
        const float* x,
        const float* y) const {
    return fvec_L2sqr(x, y, d);
}

template <>
inline float VectorDistance<METRIC_INNER_PRODUCT>::operator()(
        const float* x,
        const float* y) const {
    return fvec_inner_product(x, y, d);
}

template <>
inline float VectorDistance<METRIC_L1>::operator()(
        const float* x,
        const float* y) const {
    return fvec_L1(x, y, d);
}

template <>
inline float VectorDistance<METRIC_Linf>::operator()(
        const float* x,
        const float* y) const {
    return fvec_Linf(x, y, d);
    /*
        float vmax = 0;
        for (size_t i = 0; i < d; i++) {
            float diff = fabs (x[i] - y[i]);
            if (diff > vmax) vmax = diff;
        }
     return vmax;*/
}

template <>
inline float VectorDistance<METRIC_Lp>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    for (size_t i = 0; i < d; i++) {
        float diff = fabs(x[i] - y[i]);
        accu += powf(diff, metric_arg);
    }
    return accu;
}

template <>
inline float VectorDistance<METRIC_Canberra>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    for (size_t i = 0; i < d; i++) {
        float xi = x[i], yi = y[i];
        accu += fabs(xi - yi) / (fabs(xi) + fabs(yi));
    }
    return accu;
}

template <>
inline float VectorDistance<METRIC_BrayCurtis>::operator()(
        const float* x,
        const float* y) const {
    float accu_num = 0, accu_den = 0;
    for (size_t i = 0; i < d; i++) {
        float xi = x[i], yi = y[i];
        accu_num += fabs(xi - yi);
        accu_den += fabs(xi + yi);
    }
    return accu_num / accu_den;
}

template <>
inline float VectorDistance<METRIC_JensenShannon>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    for (size_t i = 0; i < d; i++) {
        float xi = x[i], yi = y[i];
        float mi = 0.5 * (xi + yi);
        float kl1 = -xi * log(mi / xi);
        float kl2 = -yi * log(mi / yi);
        accu += kl1 + kl2;
    }
    return 0.5 * accu;
}

template <>
inline float VectorDistance<METRIC_Jaccard>::operator()(
        const float* x,
        const float* y) const {
    // WARNING: this distance is defined only for positive input vectors.
    // Providing vectors with negative values would lead to incorrect results.
    float accu_num = 0, accu_den = 0;
    for (size_t i = 0; i < d; i++) {
        accu_num += fmin(x[i], y[i]);
        accu_den += fmax(x[i], y[i]);
    }
    return accu_num / accu_den;
}

template <>
inline float VectorDistance<METRIC_NaNEuclidean>::operator()(
        const float* x,
        const float* y) const {
    // https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html
    float accu = 0;
    size_t present = 0;
    for (size_t i = 0; i < d; i++) {
        if (!std::isnan(x[i]) && !std::isnan(y[i])) {
            float diff = x[i] - y[i];
            accu += diff * diff;
            present++;
        }
    }
    if (present == 0) {
        return NAN;
    }
    return float(d) / float(present) * accu;
}

template <>
inline float VectorDistance<METRIC_ABS_INNER_PRODUCT>::operator()(
        const float* x,
        const float* y) const {
    float accu = 0;
    for (size_t i = 0; i < d; i++) {
        accu += fabs(x[i] * y[i]);
    }
    return accu;
}

template <>
inline float VectorDistance<METRIC_HYBRID_INNER_PRODUCT>::operator()(
        const float* x,
        const float* y) const {
    
    const size_t d_dense = static_cast<size_t>(metric_arg);
    const size_t d_sparse = (d - d_dense) / 2;

    const float w_dense = 2 * (metric_arg - d_dense);
    const float w_sparse = 1 - w_dense;

    // printf("w_dense: %f, w_sparse: %f\n", w_dense, w_sparse);

    float dense_score = 0.0;
    float sparse_score = 0.0;
    
    if (w_dense > 1e-6) {
        dense_score = fvec_inner_product(x, y, d_dense);
    }
    
    if (w_sparse > 1e-6) {
        size_t x_ptr = d_dense;
        size_t y_ptr = d_dense;

        while (x_ptr < d_dense + d_sparse && y_ptr < d_dense + d_sparse) {
            float x_idx = x[x_ptr];
            float y_idx = y[y_ptr];

            float x_val = x[x_ptr + d_sparse];
            float y_val = y[y_ptr + d_sparse];

            if (x_idx == y_idx) {
                sparse_score += x_val * y_val;
                x_ptr++;
                y_ptr++;
            } else if (x_idx < y_idx) {
                x_ptr++;
            } else {
                y_ptr++;
            }
        }

        // float x_len = 0;
        // float y_len = 0;

        // for (size_t i = 0; i < d_sparse; i++) {
        //     int idx = i + d_dense + d_sparse;

        //     // L1
        //     x_len += x[idx];
        //     y_len += y[idx];

        //     // L2
        //     x_len += x[idx] * x[idx];
        //     y_len += y[idx] * y[idx];
        // }

        // // L1
        // sparse_score /= (x_len * y_len);

        // // L2
        // sparse_score /= sqrt(x_len * y_len);

        // Log normalization
        sparse_score = log(1 + sparse_score) / 4;
    }

    return w_dense * dense_score + w_sparse * sparse_score;
}

/***************************************************************************
 * Dispatching function that takes a metric type and a consumer object
 * the consumer object should contain a retun type T and a operation template
 * function f() that is called to perform the operation. The first argument
 * of the function is the VectorDistance object. The rest are passed in as is.
 **************************************************************************/

template <class Consumer, class... Types>
typename Consumer::T dispatch_VectorDistance(
        size_t d,
        MetricType metric,
        float metric_arg,
        Consumer& consumer,
        Types... args) {
    switch (metric) {
#define DISPATCH_VD(mt)                                              \
    case mt: {                                                       \
        VectorDistance<mt> vd = {d, metric_arg};                     \
        return consumer.template f<VectorDistance<mt>>(vd, args...); \
    }
        DISPATCH_VD(METRIC_INNER_PRODUCT);
        DISPATCH_VD(METRIC_L2);
        DISPATCH_VD(METRIC_L1);
        DISPATCH_VD(METRIC_Linf);
        DISPATCH_VD(METRIC_Lp);
        DISPATCH_VD(METRIC_Canberra);
        DISPATCH_VD(METRIC_BrayCurtis);
        DISPATCH_VD(METRIC_JensenShannon);
        DISPATCH_VD(METRIC_Jaccard);
        DISPATCH_VD(METRIC_NaNEuclidean);
        DISPATCH_VD(METRIC_ABS_INNER_PRODUCT);
        DISPATCH_VD(METRIC_HYBRID_INNER_PRODUCT);
        default:
            FAISS_THROW_FMT("Invalid metric %d", metric);
    }
#undef DISPATCH_VD
}

} // namespace faiss
