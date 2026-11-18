"""Serve the evaluator UI on Laptop 2:  python ui/serve.py [port]   then open http://localhost:5500
Optionally pre-fill the API address: http://localhost:5500/?api=http://<laptop1-ip>:8000"""
import functools
import http.server
import os
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5500
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.dirname(os.path.abspath(__file__)))
print(f"PathSnap UI on http://localhost:{port}  (enter the Laptop 1 API address in the page)")
http.server.ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()
