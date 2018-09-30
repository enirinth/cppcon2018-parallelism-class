/*
Copyright 2018 Gordon Brown

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __GPU_TRANSFORM_H__
#define __GPU_TRANSFORM_H__

#include <bits/sycl_policy.h>
#include <CL/sycl.hpp>
#include <iterator>

namespace cppcon {

class aaron_test;

template <class ContiguousIt, class UnaryOperation, typename KernelName>
ContiguousIt transform(sycl_execution_policy_t<KernelName> policy,
                       ContiguousIt first, ContiguousIt last,
                       ContiguousIt d_first, UnaryOperation unary_op) {
  /* implement me */
  using namespace cl::sycl;
  using value_type = typename ContiguousIt::value_type;

  queue kernelQueue;

  size_t dataSize = std::distance(first, last);

  buffer<value_type, 1> inBuf{first, last};
  buffer<value_type, 1> outBuf{d_first, d_first + dataSize};

  kernelQueue.submit([&](handler &cgh) {
    auto inputAccessor = inBuf.get_access<access::mode::read>(cgh);
    auto outputAccessor = outBuf.get_access<access::mode::write>(cgh);

    cgh.parallel_for<aaron_test>(
      range<1>{dataSize},
      [=](id<1> idx){outputAccessor[idx] = unary_op(inputAccessor[idx]);}
    );
    
  });



  return d_first;
}

}  // namespace cppcon

#endif  // __GPU_TRANSFORM_H__
