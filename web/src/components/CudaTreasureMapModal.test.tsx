import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CudaTreasureMapModal } from "./CudaTreasureMapModal";

describe("CudaTreasureMapModal", () => {
  it("renders the main CUDA optimization route labels", () => {
    render(<CudaTreasureMapModal open={true} onClose={() => undefined} />);

    expect(screen.getByRole("dialog", { name: /CUDA operator treasure route/i })).toBeInTheDocument();
    expect(screen.getByText(/Profile the Truth/i)).toBeInTheDocument();
    expect(screen.getByText(/Fix Global Access/i)).toBeInTheDocument();
    expect(screen.getByText(/Polish Instructions/i)).toBeInTheDocument();
  });
});
