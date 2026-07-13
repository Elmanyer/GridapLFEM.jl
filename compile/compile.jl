
# Execute this script to precompile the GridapSWE module and create a system image for faster loading in future runs. Command to execute:
# mpiexecjl -n 1 julia --project=. compile/compile.jl

# Note: This process may take some time as it compiles the entire module and its dependencies.
# After running this script, you can use the generated system image in your Julia sessions to speed up the loading of the GridapSWE module and its associated functions.
# To use the generated system image, start Julia with the following command:
# julia --project=. -J compile/GridapSWE_sysimage.so

using PackageCompiler

create_sysimage(:GridapSWE, 
    sysimage_path=joinpath(@__DIR__,"..","GridapSWE_sysimage.so"), 
    precompile_execution_file=joinpath(@__DIR__,"warmup.jl"))



 