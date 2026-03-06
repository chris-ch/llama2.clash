.PHONY: test test-full build run

# Fast unit tests using the nano model (~18 seconds)
test:
	cabal test llama2-test -f model-nano -f -model-260k

# Full integration tests with the 260K model (slow)
test-full:
	cabal test llama2-test

# Build the executable (260K model)
build:
	cabal build llama2

# Run the executable (260K model)
run:
	cabal run llama2
