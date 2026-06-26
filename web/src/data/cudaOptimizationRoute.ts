export type TreasureNodeKind = "main" | "branch";

export interface TreasureNode {
  id: string;
  label: string;
  kind: TreasureNodeKind;
  x: number;
  y: number;
  summary: string;
  details: string[];
  guideSections: string[];
  relatedOptimizationIds: string[];
  importance?: "normal" | "high";
  branchFrom?: string;
}

export const cudaTreasureRoute: TreasureNode[] = [
  {
    id: "profile-truth",
    label: "Profile the Truth",
    kind: "main",
    x: 10,
    y: 18,
    summary: "Measure device-side reality before changing kernel code.",
    details: [
      "Use GPU-side timing and profiler traces to find the real hotspot.",
      "Anchor optimization work to measured bottlenecks, not intuition.",
      "Capture bandwidth and latency symptoms before tuning."
    ],
    guideSections: ["4", "9"],
    relatedOptimizationIds: ["O1", "O2"],
    importance: "high"
  },
  {
    id: "guard-correctness",
    label: "Guard Correctness",
    kind: "main",
    x: 24,
    y: 30,
    summary: "Keep numerical intent and validation alongside performance work.",
    details: [
      "Check optimized kernels against trusted reference outputs.",
      "Track acceptable floating-point error bounds.",
      "Do not trade away correctness by accident."
    ],
    guideSections: ["7"],
    relatedOptimizationIds: ["O5"]
  },
  {
    id: "shape-parallel-work",
    label: "Shape Parallel Work",
    kind: "main",
    x: 38,
    y: 20,
    summary: "Expose enough parallel work before low-level tuning.",
    details: [
      "Choose decomposition that gives threads enough independent work.",
      "Prefer tuned libraries if the workload already matches them.",
      "Avoid optimizing serial structure with CUDA-specific tricks."
    ],
    guideSections: ["5", "6.1", "6.3"],
    relatedOptimizationIds: ["O3", "O4"]
  },
  {
    id: "cut-data-motion",
    label: "Cut Data Motion",
    kind: "main",
    x: 51,
    y: 36,
    summary: "Reduce host-device movement and redundant intermediate traffic.",
    details: [
      "Batch transfers and keep intermediates on GPU when possible.",
      "Treat transfer reduction as a first-class optimization target.",
      "Only optimize copy overlap after reducing copy volume."
    ],
    guideSections: ["10.1"],
    relatedOptimizationIds: ["O6"],
    importance: "high"
  },
  {
    id: "branch-streams",
    label: "Pinned / Async / Streams",
    kind: "branch",
    x: 60,
    y: 26,
    summary: "Use overlap and pinned memory when transfers are unavoidable.",
    details: [
      "Use pinned host memory for faster async copies.",
      "Overlap copies with compute through streams.",
      "Reserve this path for unavoidable transfer-heavy pipelines."
    ],
    guideSections: ["10.1.1", "10.1.2"],
    relatedOptimizationIds: ["O7", "O19"],
    branchFrom: "cut-data-motion"
  },
  {
    id: "fix-global-access",
    label: "Fix Global Access",
    kind: "main",
    x: 65,
    y: 56,
    summary: "Repair coalescing, stride, and alignment before instruction micro-tuning.",
    details: [
      "Make adjacent threads touch adjacent memory.",
      "Rewrite strided and misaligned access patterns.",
      "Use bandwidth waste as the trigger for layout changes."
    ],
    guideSections: ["10.2.1"],
    relatedOptimizationIds: ["O9", "O10"],
    importance: "high"
  },
  {
    id: "branch-l2",
    label: "L2 persistence",
    kind: "branch",
    x: 74,
    y: 46,
    summary: "Treat hot memory regions as persistent only when reuse is predictable.",
    details: [
      "Use access-policy windows for predictable reuse regions.",
      "Apply only when the workload benefits from stable cache residency."
    ],
    guideSections: ["10.2.2"],
    relatedOptimizationIds: ["O11"],
    branchFrom: "fix-global-access"
  },
  {
    id: "stage-through-shared",
    label: "Stage Through Shared",
    kind: "main",
    x: 54,
    y: 74,
    summary: "Move into shared memory only when reuse or access repair justifies it.",
    details: [
      "Tile reused inputs through shared memory.",
      "Use shared staging to repair global memory access shape.",
      "Treat shared memory as a bandwidth tool, not a default."
    ],
    guideSections: ["10.2.3.2", "10.2.3.3"],
    relatedOptimizationIds: ["O12"],
    importance: "high"
  },
  {
    id: "branch-bank-conflicts",
    label: "Bank conflicts",
    kind: "branch",
    x: 43,
    y: 84,
    summary: "Pad and lay out shared tiles to avoid conflict serialization.",
    details: [
      "Use padding in transpose-style tiles.",
      "Validate that shared-memory speedups are not lost to conflicts."
    ],
    guideSections: ["10.2.3.1", "10.2.3.3"],
    relatedOptimizationIds: ["O13"],
    branchFrom: "stage-through-shared"
  },
  {
    id: "branch-async-g2s",
    label: "Async G2S copy",
    kind: "branch",
    x: 68,
    y: 86,
    summary: "Pipeline global-to-shared movement when it helps throughput and pressure.",
    details: [
      "Use async copy paths to reduce register pressure.",
      "Overlap staging with useful work where the architecture supports it."
    ],
    guideSections: ["10.2.3.4"],
    relatedOptimizationIds: ["O14", "O15"],
    branchFrom: "stage-through-shared"
  },
  {
    id: "tune-launch-geometry",
    label: "Tune Launch Geometry",
    kind: "main",
    x: 32,
    y: 66,
    summary: "Balance threads, blocks, occupancy, registers, and shared memory together.",
    details: [
      "Tune block shape after memory behavior is understood.",
      "Do not chase occupancy blindly.",
      "Use resource balance to hide latency, not single-metric maximization."
    ],
    guideSections: ["11.1", "11.4"],
    relatedOptimizationIds: ["O17", "O18"]
  },
  {
    id: "branch-concurrency",
    label: "Concurrent kernels",
    kind: "branch",
    x: 18,
    y: 74,
    summary: "Use independent streams only when the workload really permits overlap.",
    details: [
      "Launch independent kernels concurrently when resources allow it.",
      "Prefer this for multi-stage pipelines or independent work queues."
    ],
    guideSections: ["11.5"],
    relatedOptimizationIds: ["O19"],
    branchFrom: "tune-launch-geometry"
  },
  {
    id: "polish-instructions",
    label: "Polish Instructions",
    kind: "main",
    x: 18,
    y: 48,
    summary: "Apply arithmetic and control-flow micro-optimizations after memory and launch tuning.",
    details: [
      "Reduce expensive arithmetic where accuracy allows it.",
      "Use intrinsics and precision flags deliberately.",
      "Cut divergence and use predication or unrolling when it helps."
    ],
    guideSections: ["12.1", "12.2", "13.1", "13.2"],
    relatedOptimizationIds: ["O21", "O22", "O23", "O24", "O25"]
  },
  {
    id: "branch-target-build",
    label: "Target GPU build",
    kind: "branch",
    x: 8,
    y: 60,
    summary: "Make sure optimized kernels are built for the GPU targets that matter.",
    details: [
      "Compile for relevant compute capabilities.",
      "Check deployment/runtime compatibility before trusting the optimization result."
    ],
    guideSections: ["15-18"],
    relatedOptimizationIds: ["O26"],
    branchFrom: "polish-instructions"
  }
];
