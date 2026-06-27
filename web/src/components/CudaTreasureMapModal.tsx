import { useLayoutEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { cudaTreasureMainRouteOrder, cudaTreasureRoute } from "../data/cudaOptimizationRoute";
import { CudaTreasureMap } from "./CudaTreasureMap";

interface CudaTreasureMapModalProps {
  open: boolean;
  onClose: () => void;
}

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])"
].join(", ");

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (element) => !element.hasAttribute("disabled") && element.getAttribute("aria-hidden") !== "true"
  );
}

export function CudaTreasureMapModal({ open, onClose }: CudaTreasureMapModalProps) {
  const dialogRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  const previousFocusedElementRef = useRef<HTMLElement | null>(null);

  onCloseRef.current = onClose;

  useLayoutEffect(() => {
    if (!open) {
      return undefined;
    }

    const dialog = dialogRef.current;

    if (!dialog) {
      return undefined;
    }

    previousFocusedElementRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;

    const initialFocusTarget = getFocusableElements(dialog)[0] ?? dialog;
    initialFocusTarget.focus();

    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        event.preventDefault();
        onCloseRef.current();
        return;
      }

      if (event.key !== "Tab") {
        return;
      }

      const currentDialog = dialogRef.current;

      if (!currentDialog) {
        return;
      }

      const focusableElements = getFocusableElements(currentDialog);

      if (focusableElements.length === 0) {
        event.preventDefault();
        currentDialog.focus();
        return;
      }

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];
      const activeElement = document.activeElement;

      if (!(activeElement instanceof HTMLElement) || !currentDialog.contains(activeElement)) {
        event.preventDefault();
        (event.shiftKey ? lastElement : firstElement).focus();
        return;
      }

      if (event.shiftKey && activeElement === firstElement) {
        event.preventDefault();
        lastElement.focus();
        return;
      }

      if (!event.shiftKey && activeElement === lastElement) {
        event.preventDefault();
        firstElement.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      const previousFocusedElement = previousFocusedElementRef.current;

      if (previousFocusedElement?.isConnected) {
        previousFocusedElement.focus();
      }
    };
  }, [open]);

  if (!open) {
    return null;
  }

  return createPortal(
    <div className="modal-backdrop cuda-treasure-map-modal__backdrop" role="presentation" onClick={onClose}>
      <section
        ref={dialogRef}
        className="cuda-treasure-map-modal"
        role="dialog"
        aria-modal="true"
        aria-label="CUDA operator treasure route"
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="cuda-treasure-map-modal__header">
          <div>
            <p className="panel-kicker">CUDA treasure map</p>
          </div>
          <button type="button" className="modal-close" aria-label="Close CUDA operator treasure route" onClick={onClose}>
            close
          </button>
        </header>
        <CudaTreasureMap route={cudaTreasureRoute} mainRouteOrder={cudaTreasureMainRouteOrder} />
      </section>
    </div>,
    document.body
  );
}
