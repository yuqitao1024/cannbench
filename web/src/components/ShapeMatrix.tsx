import { Fragment } from "react";
import type { ShapeAxis, ShapeStage, ShapeTensor } from "../shape-trace/types";
import { axisRatioLabel, dimensionPixels } from "../shape-trace/shapeScale";
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

function axisAccessibleName(axis: ShapeAxis): string {
  return `${formatAxis(axis)}: ${axis.meaning}`;
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
      <div className="shape-matrix-ratio">
        {axisRatioLabel({ ...tensor, axes: geometryAxes })}
      </div>
      {scopeAxes.length > 0 ? (
        <div className="shape-matrix-scopes" aria-label="Aggregate scope">
          {scopeAxes.map((axis) => (
            <span
              aria-label={axisAccessibleName(axis)}
              className="shape-matrix-scope"
              title={axis.meaning}
              key={axis.symbol}
            >
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
        aria-label={heightAxis ? axisAccessibleName(heightAxis) : "Scalar axis"}
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
        aria-label={axisAccessibleName(widthAxis)}
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
  const referencedIds = new Set([...stage.input_ids, ...stage.output_ids]);
  const supplemental = stage.tensors.filter((tensor) => !referencedIds.has(tensor.id));
  const contractedAxes = new Set(stage.contracted_axes);
  const inputSeparator = stage.operation === "matmul" ? "x" : "and";

  const operator = (value: string, key: string) => (
    <span
      className="shape-equation-operator"
      data-testid="shape-equation-operator"
      aria-hidden="true"
      key={key}
    >
      {value}
    </span>
  );

  return (
    <section className="shape-matrix-equation" aria-label={stage.title}>
      <div
        className="shape-equation-surface"
        aria-label={`${stage.title} tensor equation`}
        role="region"
        tabIndex={0}
      >
        <div className="shape-equation">
          {inputs.map((tensor, index) => (
            <Fragment key={`input-${tensor.id}`}>
              {index > 0 ? operator(inputSeparator, `input-separator-${tensor.id}`) : null}
              <ShapeMatrix tensor={tensor} contractedAxes={contractedAxes} />
            </Fragment>
          ))}
          {outputs.map((tensor, index) => (
            <Fragment key={`output-${tensor.id}`}>
              {index === 0 && inputs.length > 0
                ? operator("->", "result-transition")
                : index > 0
                  ? operator("and", `output-separator-${tensor.id}`)
                  : null}
              <ShapeMatrix
                tensor={tensor}
                contractedAxes={contractedAxes}
                variant="output"
              />
            </Fragment>
          ))}
        </div>
        {supplemental.length > 0 ? (
          <div
            className="shape-equation-supplemental"
            role="group"
            aria-label="Aggregate and supplemental tensors"
          >
            <p>Aggregate and supplemental tensors</p>
            <div className="shape-equation shape-equation-supplemental-tensors">
              {supplemental.map((tensor) => (
                <ShapeMatrix
                  tensor={tensor}
                  contractedAxes={contractedAxes}
                  key={`supplemental-${tensor.id}`}
                />
              ))}
            </div>
          </div>
        ) : null}
      </div>
      <div className="shape-equation-note">
        <strong>{stage.formula}</strong>
        <p>{stage.insight}</p>
      </div>
    </section>
  );
}
