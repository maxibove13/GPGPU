# cuda-gdb

cuda-dbg is an extension of the GNU GDB for debugging CUDA codes.

Tipically is already installed with any CUDA version.



## Initialize

First compile your program with `-g` and `-G` flags:

```nvcc -g -G my_cuda_program my_cuda_program.cu```

If only one device is present you have to enable a certain environmental variable:

```export CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1```


Then execute your program with `cuda-gdb` at the beggining:

```cuda-gdb ./my_cuda_program```

If your program has arguments pass it to `cuda-gdb`:

```r arg1 arg2```

## Set breakpoints

To set a breakpoint just type:

```break <line>```

where `<line>` is the line where you want to pause.



## Coordinates

### Hardware coordinates

You can display the hardware coordinates, device, sm (streaming multiprocessor), warp and lane.

```cuda device sm warp lane```

### Software coordinates

Analogously, you can display software coordinates: kernel, block and thread.

```cuda kernel block thread```

### Switch focus

You can also change the focus to a specific coordinate especifying the coordinates numbers.

```cuda device 0 sm 1 warp lane 3```

## Get information

### Get threads info

```info cuda threads```

## Print variables

In order to print the value of a certain variable just type:

```p my_var```

If your variable is an array, just type the position you want to print:

```p my_array[1]```

## Exit

To exit `cuda-gdb` just type `q`


# CUDA mem-check

CUDA mem-check is another CUDA utility that checks if your application has any memory errors.

To test your application with mem-check, just execute it with `mem-check` at the beggining:

```mem-check ./my_cuda_program arg1 arg2```