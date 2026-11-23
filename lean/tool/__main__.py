"""python -m lean.tool [--port 8000]: serve the web tool on localhost (needs the `tools` extra)."""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    print(f"http://127.0.0.1:{arguments.port}/")
    uvicorn.run("lean.tool.app:app", host="127.0.0.1", port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
