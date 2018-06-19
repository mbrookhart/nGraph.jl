include("../src/nGraph.jl")
using nGraph
using Base.Test

function test_abc()
    S = Shape([4,2])
    et = get_element_type(Float32(0))

    a = Parameter(et, S)
    b = Parameter(et, S)
    c = Parameter(et, S)

    d = (a + b) * c

    f = NGFunction([d], [a,b,c])

    A = Array{Float32}([1 2; 3 4; 5 6; 7 8])
    B = Array{Float32}([8 7; 6 5; 4 3; 2 1])
    C = Array{Float32}([2 2; 2 2; 2 2; 2 2])
    D = Array{Float32}([1 1; 1 1; 1 1; 1 1])

    backend = Backend("INTERPRETER")
    A_TV = TensorView(backend, A)
    B_TV = TensorView(backend, B)
    C_TV = TensorView(backend, C)
    D_TV = TensorView(backend, D)

    call(f, backend, [D_TV], [A_TV, B_TV, C_TV])

    @test read_tv(D_TV, D) ≈ (A .+ B) .* C
end

test_abc()
