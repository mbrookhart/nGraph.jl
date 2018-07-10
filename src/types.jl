import Base.convert#, Base.promote_rule

Backend = cxxt"std::shared_ptr<ngraph::runtime::Backend>"
(::Type{<:Backend})(name) = icxx"ngraph::runtime::Backend::create($name);"

ElementType = cxxt"const ngraph::element::Type &"
(::Type{<:ElementType})(::T) where {T} = icxx"ngraph::element::from<$T>();"
get_element_type(Arr) = ElementType(eltype(Arr)(0))

Shape = cxxt"std::vector< size_t >"

Parameter = cxxt"std::shared_ptr<ngraph::op::Parameter>"
(::Type{<:Parameter})(element_type::ElementType, shape::Shape) = icxx"std::make_shared<ngraph::op::Parameter>($element_type, $shape);"
ParameterVector = cxxt"std::vector< std::shared_ptr< ngraph::op::Parameter > >"

Node = cxxt"std::shared_ptr<ngraph::Node>"
NodeVector = cxxt"std::vector< std::shared_ptr< ngraph::Node > >"

convert(::Type{Node}, node::Parameter) = icxx"std::dynamic_pointer_cast<ngraph::Node>($node);"

TensorView = cxxt"std::shared_ptr<ngraph::runtime::TensorView>"
TensorViewVector = cxxt"std::vector< std::shared_ptr< ngraph::runtime::TensorView > >"

buffer_length(Arr) = icxx"$(get_element_type(Arr)).size()*$(length(Arr));"
function (::Type{<:TensorView})(backend::Backend, Arr)
    TV = icxx"($backend)->create_tensor($(get_element_type(Arr)), $(Shape([i for i=size(Arr)])));"
    icxx"($TV)->write($(pointer(Arr')), 0, $(buffer_length(Arr)));"
    TV
end
function read_tv(TV::TensorView, Arr)
    element_type = get_element_type(Arr)
    ArrT = Arr'
    icxx"($TV)->read($(pointer(ArrT)), 0, $(buffer_length(Arr)));"
    ArrT'
end

NGFunction = cxxt"std::shared_ptr<ngraph::Function>"
(::Type{<:NGFunction})(outputs::NodeVector, inputs::ParameterVector) = icxx"std::make_shared<ngraph::Function>($outputs, $inputs);"
(::Type{<:NGFunction})(outputs, inputs) = NGFunction(NodeVector(outputs), ParameterVector(inputs))
call(f::NGFunction, backend::Backend, outputs::TensorViewVector, inputs::TensorViewVector) = icxx"($backend)->call($f, $outputs, $inputs);"
call(f::NGFunction, backend::Backend, outputs, inputs) = call(f, backend, TensorViewVector(outputs), TensorViewVector(inputs))

export Shape, ElementType, get_element_type, Backend, NGFunction, call, Node, Parameter,
       NodeVector, ParameterVector, TensorView, TensorViewVector, read_tv, write_tv
