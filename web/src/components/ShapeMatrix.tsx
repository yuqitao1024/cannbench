import { Fragment } from "react";
import type { ShapeAxis, ShapeStage, ShapeTensor } from "../shape-trace/types";
import { dimensionPixels } from "../shape-trace/shapeScale";
import "../shape-explorer.css";

interface ShapeMatrixProps {
  tensor: ShapeTensor;
  contractedAxes: ReadonlySet<string>;
  variant?: "input" | "output";
}

interface ShapeMatrixEquationProps {
  stage: ShapeStage;
}

function formatAxis(axis: ShapeAxis): string {
  return `${axis.symbol}=${axis.value.toLocaleString("en-US")}`;
}

function axisClassName(baseClass: string, axis: ShapeAxis, contractedAxes: ReadonlySet<string>) {
  return contractedAxes.has(axis.symbol) ? `${baseClass} contracted-axis` : baseClass;
}

export function ShapeMatrix({
  tensor,
  contractedAxes,
  variant = "input"
}: ShapeMatrixProps) {
  const geometryAxes = tensor.axes.slice(-2);
  const widthAxis = geometryAxes.at(-1)!;
  const heightAxis = geometryAxes.length === 1 ? null : geometryAxes[0];
  const scopeAxes = tensor.axes.slice(0, -2);
  const width = dimensionPixels(widthAxis.value);
  const height = heightAxis ? dimensionPixels(heightAxis.value) : 36;

  return (
    <figure
      className={`shape-matrix shape-matrix-${variant}`}
      aria-label={`${tensor.label} tensor`}
    >
      <figcaption className="shape-matrix-caption">
        <span>{tensor.label}</span>
        {tensor.logical_only ? <small>logical view</small> : null}
      </figcaption>
      {scopeAxes.length > 0 ? (
        <div className="shape-matrix-scopes" aria-label="Aggregate scope">
          {scopeAxes.map((axis) => (
            <span className="shape-matrix-scope" title={axis.meaning} key={axis.symbol}>
              {formatAxis(axis)}
            </span>
          ))}
        </div>
      ) : null}
      <div
        className={
          heightAxis
            ? axisClassName("shape-matrix-y", heightAxis, contractedAxes)
            : "shape-matrix-y"
        }
        title={heightAxis?.meaning}
      >
        {heightAxis ? formatAxis(heightAxis) : "1"}
      </div>
      <div
        className="shape-matrix-body"
        data-testid={`matrix-${tensor.id}`}
        data-rank={tensor.axes.length}
        style={{ width: `${width}px`, height: `${height}px` }}
      />
      <div
        className={axisClassName("shape-matrix-x", widthAxis, contractedAxes)}
        title={widthAxis.meaning}
      >
        {formatAxis(widthAxis)}
      </div>
    </figure>
  );
}

function tensorsFor(ids: string[], tensorsById: ReadonlyMap<string, ShapeTensor>) {
  return ids.flatMap((id) => {
    const tensor = tensorsById.get(id);
    return tensor ? [tensor] : [];
  });
}

export function ShapeMatrixEquation({ stage }: ShapeMatrixEquationProps) {
  const tensorsById = new Map(stage.tensors.map((tensor) => [tensor.id, tensor]));
  const inputs = tensorsFor(stage.input_ids, tensorsById);
  const outputs = tensorsFor(stage.output_ids, tensorsById);
  const contractedAxes = new Set(stage.contracted_axes);

  return (
    <section className="shape-matrix-equation" aria-label={stage.title}>
      <div className="shape-equation-surface">
        <div className="shape-equation">
          {inputs.map((tensor, index) => (
            <Fragment key={`input-${tensor.id}`}>
              {stage.operation === "matmul" && index > 0 ? (
                <span className="shape-equation-operator" aria-hidden="true">
                  x
                </span>
              ) : null}
              <ShapeMatrix tensor={tensor} contractedAxes={contractedAxes} />
            </Fragment>
          ))}
          {outputs.map((tensor, index) => (
            <Fragment key={`output-${tensor.id}`}>
              {inputs.length > 0 || index > 0 ? (
                <span className="shape-equation-operator" aria-hidden="true">
                  -&gt;
                </span>
              ) : null}
              <ShapeMatrix
                tensor={tensor}
                contractedAxes={contractedAxes}
                variant="output"
              />
            </Fragment>
          ))}
        </div>
      </div>
      <div className="shape-equation-note">
        <strong>{stage.formula}</strong>
        <p>{stage.insight}</p>
      </div>
    </section>
  );
}
