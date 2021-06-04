Copy Task
=========

In this task, the transformer learns to copy a random sequence of symbols. This
benchmark requires very sparse random attention and is a hard task for dense
attentions like linear attention.

The transformer reads, one element at a time, a sequence like the following

    0 t1 t2 t3 t4 0 t1 t2 t3 t4

and must predict the next element. The loss is taken in the second half of the
sequence as t1-t4 are completely random tokens *but* they are repeated.

Requirements
------------

The installation requirements are the following:

* torch
* pytorch-fast-transformers

They can be installed in most systems via

    pip install torch pytorch-fast-transformers

Running the code
----------------

The `main.py` has thorough command line help that can be invoked via the
`--help` command line argument. The attention implementation is chosen via the
`--attention_type` argument.

    python main.py  # Runs with the causal-linear attention_type
    python main.py --attention_type full  # Uses softmax attention
    python main.py --attention_type reformer  # Uses reformer or lsh attention
                                              # see also --rounds for changing the
                                              # hashing rounds
                                              
Additionally, one can repeat the experiments from this paper by manipulating the
following flexible command line arguments:

    python main.py --attention_type FMM --kernels 3 # Uses rank 3 attention
    python main.py --attention_type FMM --sparse --bandwidth 10 # Uses linear attention + bw 10
    python main.py --attention_type FMM --kernels 3 --sparse --bandwidth 10 # Uses rank 3 attention + bw 10

Other things to try are increasing the maximum length of the sequence

    python main.py --sequence_length 512

or playing around with the transformer parameters like `--rounds` and `--bits`
for LSH attention.
