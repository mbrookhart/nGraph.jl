include("../src/nGraph.jl")
using nGraph
using Base.Test

backend_name = "INTERPRETER"

function test_unary(func)
    S = Shape([4,2])
    et = get_element_type(Float32(0))
    a = Parameter(et, S)
    b = func(a)

    f = NGFunction([b], [a])

    A = rand(Float32, (4,2))
    if (func in [relu, abs])
        A .-= 0.5
    end
    B = ones(A)

    backend = Backend(backend_name)
    A_TV = TensorView(backend, A)
    B_TV = TensorView(backend, B)

    call(f, backend, [B_TV], [A_TV])
    @test read_tv(B_TV, B) ≈ func.(A)
end

test_unary(abs)
test_unary(acos)
test_unary(asin)
test_unary(atan)
test_unary(cosh)
test_unary(cos)
test_unary(exp)
test_unary(log)
test_unary(-)
test_unary(relu)
test_unary(sinh)
test_unary(sin)
test_unary(sqrt)
test_unary(tanh)
test_unary(tan)

function test_binary(func)
    S = Shape([4,2])
    et = get_element_type(Float32(0))
    a = Parameter(et, S)
    b = Parameter(et, S)
    c = func(a,b)

    f = NGFunction([c], [a,b])

    A = rand(Float32, (4,2))
    B = rand(Float32, (4,2))

    C = ones(A)

    backend = Backend(backend_name)
    A_TV = TensorView(backend, A)
    B_TV = TensorView(backend, B)
    C_TV = TensorView(backend, C)

    call(f, backend, [C_TV], [A_TV, B_TV])
    @test read_tv(C_TV, C) ≈ func.(A, B)
end

test_binary(+)
test_binary(-)
test_binary(*)
test_binary(/)
# test_binary(&)

function test_abc()
    S = Shape([4,2])
    et = get_element_type(Float32(0))

    a = Parameter(et, S)
    b = Parameter(et, S)
    c = Parameter(et, S)

    d = (a + b) * c

    f = NGFunction([d], [a,b,c])

    A = rand(Float32, (4,2))
    B = rand(Float32, (4,2))
    C = rand(Float32, (4,2))
    D = ones(C)

    backend = Backend(backend_name)
    A_TV = TensorView(backend, A)
    B_TV = TensorView(backend, B)
    C_TV = TensorView(backend, C)
    D_TV = TensorView(backend, D)

    call(f, backend, [D_TV], [A_TV, B_TV, C_TV])

    @test read_tv(D_TV, D) ≈ (A .+ B) .* C
end

test_abc()
