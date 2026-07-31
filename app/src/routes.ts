/** App-level routing. Lives here rather than inside TopNav so screens that don't
 *  render TopNav (Workbench) aren't importing a nav component just for a type. */
export type Route = "dashboard" | "workbench" | "reveal";
