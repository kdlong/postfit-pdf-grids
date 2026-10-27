import argparse
import logging
import sys

import numpy as np

from postfit_pdf_grids.covariance import read_covariance_file
from postfit_pdf_grids.postfit import PostfitPdf


def _setup_logging(verbose):
    levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ]
    logging.basicConfig(
        level=levels[verbose], format="%(levelname)s %(name)s: %(message)s"
    )


def default_set_name(pdf_name, label, scale):
    scale_label = "unscaled" if scale == 1 else f"uncx{scale:.1f}".replace(".", "p")
    return f"{pdf_name}_{label}_{scale_label}"


def make_grids(argv=None):
    parser = argparse.ArgumentParser(
        description="Write a postfit LHAPDF set from a postfit PDF covariance file."
    )
    parser.add_argument(
        "-f", "--input", required=True, help="Postfit PDF covariance HDF5 file"
    )
    parser.add_argument(
        "-o",
        "--outfolder",
        required=True,
        help="Output folder for the new set (created if needed)",
    )
    parser.add_argument(
        "-i", "--lhaid", required=True, help="LHAPDF ID to give the new set"
    )
    parser.add_argument(
        "-l",
        "--fitLabel",
        default="postfit",
        help="Label in the default set name, <pdf>_<label>_<scale>",
    )
    parser.add_argument(
        "-n", "--setName", default=None, help="Set name, overriding the default"
    )
    parser.add_argument(
        "--description",
        default=None,
        help="SetDesc of the new set; the command is appended",
    )
    parser.add_argument(
        "-v", "--verbose", type=int, choices=range(5), default=3, help="Verbosity"
    )
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    postfit = PostfitPdf.from_file(args.input)
    set_name = args.setName or default_set_name(
        postfit.pdf_name, args.fitLabel, postfit.scale
    )
    description = (
        args.description
        or f"{postfit.pdf_name} constrained by a postfit covariance, prefit uncertainties scaled by {postfit.scale}"
    )
    description += f". Produced by the command {' '.join(sys.argv)}"
    postfit.write(args.outfolder, set_name, args.lhaid, description)


def print_covariance(argv=None):
    parser = argparse.ArgumentParser(
        description="Print the content of a postfit PDF covariance file."
    )
    parser.add_argument("input", help="Postfit PDF covariance HDF5 file")
    args = parser.parse_args(argv)

    content = read_covariance_file(args.input)
    print(f"pdf_name: {content['pdf_name']}")
    print(f"pdf_scale: {content['pdf_scale']}")
    for key, value in content["metadata"].items():
        print(f"{key}: {value}")
    print("labels:", content["labels"])
    print("pulls:", content["pulls"])
    print("covariance shape:", content["covariance"].shape)
    print("covariance diagonal:", np.diag(content["covariance"]))
    print("member_weights shape:", content["member_weights"].shape)
