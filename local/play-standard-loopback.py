"""Run mjlab's unmodified play workflow and UI, binding only to localhost.

The only adapter is the supported ViserPlayViewer(viser_server=...) argument.
No custom widgets, markers, policy loop or checkpoint handling are added.
"""
from functools import partial

import viser
from mjlab.scripts import play


def main():
    server = viser.ViserServer(host='127.0.0.1', port=8086, label='mjlab')
    play.ViserPlayViewer = partial(play.ViserPlayViewer, viser_server=server)
    try:
        play.main()
    finally:
        server.stop()


if __name__ == '__main__':
    main()
