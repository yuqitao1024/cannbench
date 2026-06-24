import type { OperatorSummary } from "../types";

interface OperatorRailProps {
  operators: OperatorSummary[];
  selectedOperator: string;
  search: string;
  onSearchChange: (value: string) => void;
  onSelectOperator: (operator: string) => void;
}

export function OperatorRail({
  operators,
  selectedOperator,
  search,
  onSearchChange,
  onSelectOperator
}: OperatorRailProps) {
  const filtered = operators.filter((operator) =>
    operator.name.toLowerCase().includes(search.trim().toLowerCase())
  );

  return (
    <aside className="operator-rail" aria-label="Operators">
      <div className="rail-head">
        <p className="panel-kicker">Operators</p>
        <input
          aria-label="Search operators"
          className="operator-search"
          value={search}
          placeholder="filter ops"
          onChange={(event) => onSearchChange(event.target.value)}
        />
      </div>
      <div className="operator-list">
        {filtered.map((operator) => (
          <button
            key={operator.name}
            type="button"
            className="operator-button"
            aria-pressed={operator.name === selectedOperator}
            onClick={() => onSelectOperator(operator.name)}
          >
            <span>{operator.name}</span>
            <small>
              {operator.caseCount} cases / {operator.recordCount} rows
            </small>
          </button>
        ))}
      </div>
    </aside>
  );
}
