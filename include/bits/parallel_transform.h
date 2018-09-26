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

#ifndef __PARALLEL_TRANSFORM_H__
#define __PARALLEL_TRANSFORM_H__

#include <functional>
#include <iterator>
#include <thread>
#include <vector>

#include <bits/std_policies.h>

namespace cppcon {

template <class ForwardIt, class OutputIt, class UnaryOperation>
OutputIt transform(par_execution_policy_t policy, ForwardIt first,
                   ForwardIt last, OutputIt d_first, UnaryOperation unary_op) {
  using diff_t = typename std::iterator_traits<ForwardIt>::difference_type;

  /* If the input range is empty then return the output iterator */
  if (first == last) return d_first;

  /* Retrieve the hardware concurrency of you system */
  unsigned int concurrency = std::thread::hardware_concurrency();

  /* Calculate the data size, the base chunk size and the remainder */
  diff_t dataSize = std::distance(first, last);
  diff_t baseChunkSize = dataSize / concurrency;
  diff_t remainder = dataSize % concurrency;

  /* Create a lambda function for processing a chunk */
  auto processChunk = [unary_op](ForwardIt first, ForwardIt d_first,
                                 OutputIt last) {
    /* Iterate over the input range and output range */
    for (; first != last; ++first, ++d_first) {
      /* Read the value of the input iterator, pass it to the unary operator and
       * write the result to the output iterator */
      *d_first = unary_op(*first);
    }
  };

  /* Reserve a vector of threads for concurrency - 1 */
  std::vector<std::thread> threads(0);
  threads.reserve(concurrency - 1);

  /* Loop over the hardware concurrency, starting at 1 */
  for (unsigned int t = 1; t < concurrency; t++) {
    /* Calcualte the current chunk size as the base chunk size plus a potential
     * remaider */
    diff_t currentChunkSize =
        baseChunkSize + static_cast<diff_t>(t < remainder);

    /* Calculate the offset to the current chunk */
    diff_t currentOffset =
        (t * baseChunkSize) + std::min(static_cast<diff_t>(t), remainder);

    /* Create iterators to the beggining and end of the current chunk */
    auto chunkFirst = std::next(first, currentOffset);
    auto chunkDFirst = std::next(d_first, currentOffset);
    auto chunkLast = std::next(first, currentOffset + currentChunkSize);

    /* Launch a std::thread that will process a chunk with the iterators created
     * above*/
    threads.emplace_back(
        [=]() mutable { processChunk(chunkFirst, chunkDFirst, chunkLast); });
  }

  /* Calculate the chunk size for the chunk that will execute on the calling
   * thread and create an iterator to the end of that chunk */
  diff_t currentChunkSize = baseChunkSize + static_cast<diff_t>(0 < remainder);
  auto chunkLast = std::next(first, currentChunkSize);

  /* Process the chunk of the calling thread  */
  processChunk(first, d_first, chunkLast);

  /* Join all of the threads you created */
  for (auto &thread : threads) {
    thread.join();
  }

  /* Return the output iterator */
  return d_first;
}

}  // namespace cppcon

#endif  // __PARALLEL_TRANSFORM_H__
