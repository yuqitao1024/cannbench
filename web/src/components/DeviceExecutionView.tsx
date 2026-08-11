import type { DeviceKernelTrace, ShapeAxis, ShapeTrace } from "../shape-trace/types";

interface DeviceExecutionViewProps {
  execution: ShapeTrace["device_execution"];
}

const MAX_REPRESENTATIVE_LANES = 8;

function implementationLabel(implementation: string, version: string | null): string {
  const name = implementation.toUpperCase();
  return version ? `${name} ${version}` : name;
}

function taskAxisLabel(axis: ShapeAxis, taskIndex: number, stride: number): string {
  const value = Math.floor(taskIndex / stride) % Math.max(axis.value, 1);
  return `${axis.symbol}${value}`;
}

function representativeTasks(kernel: DeviceKernelTrace): string[] {
  const count = Math.min(kernel.task_count, MAX_REPRESENTATIVE_LANES);
  return Array.from({ length: count }, (_, taskIndex) => {
    if (kernel.task_axes.length === 0) return `Task ${taskIndex + 1}`;
    return kernel.task_axes
      .map((axis, axisIndex) => {
        const stride = kernel.task_axes
          .slice(axisIndex + 1)
          .reduce((product, laterAxis) => product * Math.max(laterAxis.value, 1), 1);
        return taskAxisLabel(axis, taskIndex, stride);
      })
      .join(" / ");
  });
}

function tensorShape(kernel: DeviceKernelTrace, tensorIndex: number): string {
  return `[${kernel.tile_tensors[tensorIndex].axes.map((axis) => axis.value).join(",")}]`;
}

export function DeviceExecutionView({ execution }: DeviceExecutionViewProps) {
  if (execution.status === "unavailable") {
    if (execution.message === null) return null;
    return (
      <section className="shape-device-unavailable" role="status">
        <p>{execution.message}</p>
      </section>
    );
  }

  return (
    <section className="shape-device-view" aria-label="Current device execution">
      <header className="shape-device-header">
        <div>
          <p className="shape-section-label">Current device execution</p>
          <h2>{implementationLabel(execution.implementation, execution.version)}</h2>
        </div>
        <span>{execution.kernels.length} kernels</span>
      </header>
      <div className="shape-device-kernels">
        {execution.kernels.map((kernel) => {
          const lanes = representativeTasks(kernel);
          return (
            <article className="shape-kernel-row" data-kernel-id={kernel.id} key={kernel.id}>
              <div className="shape-kernel-meta">
                <h3>{kernel.title}</h3>
                <p>{kernel.summary}</p>
                <dl>
                  <div>
                    <dt>Tasks</dt>
                    <dd>{kernel.task_count}</dd>
                  </div>
                  <div>
                    <dt>Used cores</dt>
                    <dd>{kernel.used_core_count}</dd>
                  </div>
                  <div>
                    <dt>Formula</dt>
                    <dd>{kernel.task_formula}</dd>
                  </div>
                </dl>
              </div>
              <div className="shape-task-lane">
                <div className="shape-task-samples" aria-label={`${kernel.title} representative tasks`}>
                  {lanes.map((lane, index) => (
                    <span className="shape-task-block" key={`${kernel.id}-${index}`}>
                      {lane}
                    </span>
                  ))}
                  {kernel.task_count > lanes.length ? (
                    <span className="shape-task-remainder">
                      +{kernel.task_count - lanes.length} tasks
                    </span>
                  ) : null}
                </div>
                <div className="shape-tile-sequence" aria-label={`${kernel.title} tile tensors`}>
                  {kernel.tile_tensors.map((tensor, index) => (
                    <span className="shape-tile-group" key={tensor.id}>
                      {index > 0 ? <span className="shape-tile-arrow">-&gt;</span> : null}
                      <span className="shape-tile-chip">
                        <span>{tensor.label}</span>
                        <strong>{tensorShape(kernel, index)}</strong>
                        {tensor.logical_only ? <small>logical</small> : null}
                      </span>
                    </span>
                  ))}
                </div>
                <ol className="shape-kernel-steps" aria-label={`${kernel.title} ordered steps`}>
                  {kernel.steps.map((step, index) => (
                    <li key={`${kernel.id}-step-${index}`}>{step}</li>
                  ))}
                </ol>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
