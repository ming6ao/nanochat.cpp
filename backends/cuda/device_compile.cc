// backends/cuda/device_compile.cc -- the whole device backend, linked once.
//
// Nothing here runs in a product. `//backends/cuda:device_compile` exists so
// that one build compiles and links every device source of the backend for the
// architecture and the precision of the configuration.
// `tools/nanochat test --profile <name>` builds it before it runs a simulator
// suite, so the instantiation of the card the run answers as is compiled, and
// not only the architecture of the default build.
//
// The link is the mechanism: every archive of the backend is an input of the
// link action, and every archive needs its objects compiled. The `main` below
// references nothing, so no kernel object is pulled into the executable and an
// unresolved seam symbol cannot fail the link.

int main() { return 0; }
