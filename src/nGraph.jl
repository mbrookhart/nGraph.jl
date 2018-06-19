module nGraph

ENV["JULIA_CXX_RTTI"] = 1
using Cxx

const ngraph_dist = "/code/ngraph_dist"
const ngraph_include_dir = joinpath(ngraph_dist, "include")

addHeaderDir(ngraph_include_dir, kind=C_User)
ENV["LD_LIBRARY_PATH"] = joinpath(ngraph_dist, "lib")

Libdl.dlopen(joinpath(ngraph_dist, "lib", "libngraph.so"), Libdl.RTLD_GLOBAL)

cxxinclude("ngraph/ngraph.hpp")
cxxinclude("vector")
cxxinclude("set")

include("types.jl")

end
