# DESIGN_TODO

A terse catalogue of open tasks only. Gaps whose fix sits in glass2glass are
tracked in that repo's `design/TODO.md`, not here.

## g2g backend coverage

- **`g2g-mcp` takes a g2g-spelled pipeline only**, so a gst-spelled one has to
  go through the launcher's rewrite before it can be started there.

- **A hosted element's properties are only checked once its class loads.** The
  g2g host takes any name it does not read itself and hands it to the Python
  class, which is the only thing that knows the real set, so a typo fails at
  pipeline start rather than at parse. `gst-inspect` on `pyelement` lists the
  host's own properties and says the rest come from the class.
