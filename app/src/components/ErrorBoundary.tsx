import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/** No error boundary existed anywhere in the tree — an uncaught exception (e.g.
 * inside ModelViewer's three.js effects) blanked the whole app with a white
 * screen and no way back except a full restart. This gives the user a recovery
 * path instead. React error boundaries must be class components; there is no
 * hook equivalent. */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[ARGUS] Uncaught error in the UI tree:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100vh",
            gap: 12,
            fontFamily: "system-ui, sans-serif",
            color: "#e5e5e5",
            background: "#0a0a0a",
            textAlign: "center",
            padding: 24,
          }}
        >
          <h2 style={{ margin: 0 }}>Something went wrong</h2>
          <p style={{ margin: 0, opacity: 0.7, maxWidth: 480, fontSize: 13 }}>
            {this.state.error.message || "The interface hit an unexpected error."}
          </p>
          <button
            onClick={() => this.setState({ error: null })}
            style={{
              marginTop: 8,
              padding: "8px 16px",
              borderRadius: 6,
              border: "1px solid #333",
              background: "#1a1a1a",
              color: "#e5e5e5",
              cursor: "pointer",
            }}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
