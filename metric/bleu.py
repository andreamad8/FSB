import itertools
import numpy as np
import numpy
import os
import re
import json
import subprocess
import tempfile
import numpy as np
from collections import Counter
from six.moves import urllib
import torch
import torch.nn as nn
import torch.nn.functional as F


# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BLEU metric implementation.
"""


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    multi_bleu_path = "metric/multi-bleu.perl"
    os.chmod(multi_bleu_path, 0o755)


    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

     # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = 0.0

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score



