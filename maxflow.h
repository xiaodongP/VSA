#ifndef MAXFLOW_HEADER
#define MAXFLOW_HEADER

#include <vector>
#include <queue>
#include <algorithm>
#include <limits>

// Edmonds-Karp maxflow (BFS-based Ford-Fulkerson).
// Reliable and sufficient for graph cut boundary smoothing
// where graphs are small (k-ring faces around boundaries).

template <typename CapType>
class MaxFlow {
public:
    enum { SOURCE = 0, SINK = 1 };

    int add_node(int num = 1) {
        int i = (int)adj_.size();
        adj_.resize(i + num);
        return i;
    }

    void add_tedge(int i, CapType cap_s, CapType cap_t) {
        // Source edge: node i <-> source (virtual node index = N)
        // Sink edge: node i <-> sink (virtual node index = N+1)
        // We encode source/sink as t-edges using excess
        source_cap_.resize(adj_.size(), 0);
        sink_cap_.resize(adj_.size(), 0);
        source_cap_[i] += cap_s;
        sink_cap_[i] += cap_t;
    }

    void add_edge(int i, int j, CapType cap, CapType rev_cap) {
        int fi = (int)edges_.size();
        edges_.push_back({j, cap, fi + 1});
        edges_.push_back({i, rev_cap, fi});
        adj_[i].push_back(fi);
        adj_[j].push_back(fi + 1);
    }

    CapType maxflow() {
        int N = (int)adj_.size();
        // Augment s->t using t-edges
        for (int i = 0; i < N; i++) {
            CapType push = std::min(source_cap_[i], sink_cap_[i]);
            if (push > 0) {
                flow_ += push;
                source_cap_[i] -= push;
                sink_cap_[i] -= push;
            }
        }

        // Main loop: BFS to find augmenting paths through the graph
        // with s-t connectivity via source_cap_[i] > 0 and sink_cap_[j] > 0
        edge_to_.assign(N, -1);
        while (true) {
            // BFS
            vector<int> parent(N, -1); // -1 = unvisited
            // Nodes 0..N-1 = mesh nodes, N = source, N+1 = sink (virtual)
            // Use separate encoding: for each mesh node i,
            // we treat source_cap_[i] > 0 as a residual edge from source,
            // sink_cap_[i] > 0 as a residual edge to sink.
            queue<int> q;
            // Start from all nodes with source excess
            for (int i = 0; i < N; i++) {
                if (source_cap_[i] > 0) {
                    q.push(i);
                    parent[i] = N; // parent = source (sentinel)
                }
            }
            int sink_node = -1;
            while (!q.empty() && sink_node < 0) {
                int u = q.front(); q.pop();
                // Check if this node can reach sink directly
                if (sink_cap_[u] > 0) {
                    sink_node = u;
                    break;
                }
                // Traverse graph edges
                for (int ei : adj_[u]) {
                    if (edges_[ei].cap <= 0) continue;
                    int v = edges_[ei].to;
                    if (parent[v] >= 0) continue;
                    parent[v] = u;
                    edge_to_[v] = ei;
                    // Also store the edge index for backtracking
                    // (We already use edge_to_ for this)
                    q.push(v);
                }
            }

            if (sink_node < 0) break; // No augmenting path

            // Find bottleneck
            CapType bottleneck = sink_cap_[sink_node];
            // Also include source cap at the origin
            {
                int n = sink_node;
                while (parent[n] != N) {
                    bottleneck = std::min(bottleneck, edges_[edge_to_[n]].cap);
                    n = parent[n];
                }
                bottleneck = std::min(bottleneck, source_cap_[n]);
            }

            if (bottleneck <= 0) break;

            // Push flow
            {
                int n = sink_node;
                // Push to sink
                sink_cap_[n] -= bottleneck;
                // Trace back to source
                while (parent[n] != N) {
                    int ei = edge_to_[n];
                    edges_[ei].cap -= bottleneck;
                    edges_[ei ^ 1].cap += bottleneck;
                    n = parent[n];
                }
                source_cap_[n] -= bottleneck;
            }
            flow_ += bottleneck;
        }

        return flow_;
    }

    int what_segment(int i) const {
        // BFS/DFS from source to determine reachability
        int N = (int)adj_.size();
        if (source_cap_[i] > 0) return SOURCE;
        // Check reachability from any source node
        vector<bool> visited(N, false);
        queue<int> q;
        for (int j = 0; j < N; j++) {
            if (source_cap_[j] > 0) { q.push(j); visited[j] = true; }
        }
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int ei : adj_[u]) {
                if (edges_[ei].cap <= 0) continue;
                int v = edges_[ei].to;
                if (visited[v]) continue;
                visited[v] = true;
                q.push(v);
            }
        }
        return visited[i] ? SOURCE : SINK;
    }

private:
    struct Edge {
        int to;
        CapType cap;
        int sister;
    };

    vector<vector<int>> adj_;     // adjacency lists of edge indices
    vector<Edge> edges_;
    vector<CapType> source_cap_;  // t-edge capacity to source
    vector<CapType> sink_cap_;    // t-edge capacity to sink
    vector<int> edge_to_;         // for BFS backtracking

    CapType flow_ = 0;
};

#endif
