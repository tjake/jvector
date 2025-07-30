/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public class ImmutablePQVectors extends PQVectors {
    private final int vectorCount;
    private final VectorFloat<?> codebookPartialSums;

    /**
     * Construct an immutable PQVectors instance with the given ProductQuantization and compressed data chunks.
     * @param pq the ProductQuantization to use
     * @param compressedDataChunks the compressed data chunks
     * @param vectorCount the number of vectors
     * @param vectorsPerChunk the number of vectors per chunk
     */
    public ImmutablePQVectors(ProductQuantization pq, ByteSequence<?>[] compressedDataChunks, int vectorCount, int vectorsPerChunk) {
        super(pq);
        this.compressedDataChunks = compressedDataChunks;
        this.vectorCount = vectorCount;
        this.vectorsPerChunk = vectorsPerChunk;
        this.codebookPartialSums = pq.createCodebookPartialSums();
    }

    @Override
    protected int validChunkCount() {
        return compressedDataChunks.length;
    }

    @Override
    public int count() {
        return vectorCount;
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction diversityFunctionFor(int node1, VectorSimilarityFunction similarityFunction) {
        final int subspaceCount = pq.getSubspaceCount();

        if (true)
            return super.diversityFunctionFor(node1, similarityFunction);

        var node1Chunk = getChunk(node1);
        var node1Offset = getOffsetInChunk(node1);

        int clusterCount = pq.getClusterCount();


        switch (similarityFunction) {
            case DOT_PRODUCT:
                return (node2) -> {
                    var node2Chunk = getChunk(node2);
                    var node2Offset = getOffsetInChunk(node2);
                    // compute the euclidean distance between the query and the codebook centroids corresponding to the encoded points
                    float dp = 0;
                    for (int m = 0; m < subspaceCount; m++) {
                        int centroidIndex1 = Byte.toUnsignedInt(node1Chunk.get(m + node1Offset));
                        int centroidIndex2 = Byte.toUnsignedInt(node2Chunk.get(m + node2Offset));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        dp += VectorUtil.dotProduct(pq.codebooks[m], centroidIndex1 * centroidLength, pq.codebooks[m], centroidIndex2 * centroidLength, centroidLength);
                    }
                    // scale to [0, 1]
                    return (1 + dp) / 2;
                };
            case COSINE:
                float norm1 = 0.0f;
                for (int m1 = 0; m1 < subspaceCount; m1++) {
                    int centroidIndex = Byte.toUnsignedInt(node1Chunk.get(m1 + node1Offset));
                    int centroidLength = pq.subvectorSizesAndOffsets[m1][0];
                    var codebookOffset = centroidIndex * centroidLength;
                    norm1 += VectorUtil.dotProduct(pq.codebooks[m1], codebookOffset, pq.codebooks[m1], codebookOffset, centroidLength);
                }
                final float norm1final = norm1;
                return (node2) -> {
                    var node2Chunk = getChunk(node2);
                    var node2Offset = getOffsetInChunk(node2);
                    // compute the dot product of the query and the codebook centroids corresponding to the encoded points
                    float sum = 0;
                    float norm2 = 0;
                    for (int m = 0; m < subspaceCount; m++) {
                        int centroidIndex1 = Byte.toUnsignedInt(node1Chunk.get(m + node1Offset));
                        int centroidIndex2 = Byte.toUnsignedInt(node2Chunk.get(m + node2Offset));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        int codebookOffset = centroidIndex2 * centroidLength;
                        sum += VectorUtil.dotProduct(pq.codebooks[m], codebookOffset, pq.codebooks[m], centroidIndex1 * centroidLength, centroidLength);
                        norm2 += VectorUtil.dotProduct(pq.codebooks[m], codebookOffset, pq.codebooks[m], codebookOffset, centroidLength);
                    }
                    float cosine = sum / (float) Math.sqrt(norm1final * norm2);
                    // scale to [0, 1]
                    return (1 + cosine) / 2;
                };
            case EUCLIDEAN:
                return (node2) -> {
                    var node2Chunk = getChunk(node2);
                    var node2Offset = getOffsetInChunk(node2);
                    // compute the euclidean distance between the query and the codebook centroids corresponding to the encoded points
                    float sum = VectorUtil.assembleAndSum2(codebookPartialSums, subspaceCount, node1Chunk, node1Offset, node2Chunk, node2Offset, clusterCount);

                    // scale to [0, 1]
                    return 1 / (1 + sum);
                };
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }
}
