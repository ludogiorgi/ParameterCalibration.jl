#!/usr/bin/env julia
# Regenerate repository dependency diagrams (Mermaid + GraphViz DOT)
# - Scans files under src/ (and optionally examples/, systems/ for example edges)
# - Extracts: module declarations, include(...) edges, using/import targets
# - Emits:
#   docs/diagrams/dependencies.mmd
#   docs/diagrams/dependencies.dot

using Printf
using Dates

struct Node
    id::String      # unique id (path or module)
    label::String   # display label
    kind::Symbol    # :file | :module | :external
end

struct Edge
    src::String
    dst::String
    kind::Symbol    # :include | :using
end

function read_lines(path)
    try
        return readlines(path)
    catch
        return String[]
    end
end

function find_src_files()
    files = String[]
    for (root, _, fs) in walkdir("src")
        for f in fs
            endswith(f, ".jl") || continue
            push!(files, joinpath(root, f))
        end
    end
    sort!(files)
    return files
end

function parse_file(path::String)
    lines = read_lines(path)
    modname = nothing
    includes = String[]
    usings = Vector{Vector{String}}()
    for line in lines
        s = strip(line)
        isempty(s) && continue
        # strip line comments
        if startswith(s, "#"); continue; end
        # module Foo
        if startswith(s, "module ")
            m = match(r"^module\s+([A-Za-z_][A-Za-z0-9_]*)", s)
            m !== nothing && (modname = String(m.captures[1]))
        end
        # include("file.jl") possibly with joinpath-like simple strings
        if startswith(s, "include(")
            q1 = findfirst('"', s)
            q1 === nothing && continue
            q2 = findnext('"', s, q1+1)
            q2 === nothing && continue
            inc = s[q1+1:q2-1]
            push!(includes, String(inc))
        end
        # using A, B: c, d or using A; import B
        if startswith(s, "using ") || startswith(s, "import ")
            m = match(r"^(using|import)\s+(.+)$", s)
            m === nothing && continue
            tail = strip(replace(m.captures[2], ";" => " "))
            # split by commas
            parts = split(tail, ',')
            mods = String[]
            for part in parts
                p = strip(part)
                isempty(p) && continue
                startswith(p, ".") && continue
                # cut off trailing colon clauses:  A: x, y => keep A
                p = split(p, ':')[1]
                # take top-level module before dot
                p = split(p, '.')[1]
                # take first token in case of 'A B'
                p = split(p)[1]
                isempty(p) && continue
                push!(mods, p)
            end
            !isempty(mods) && push!(usings, unique(mods))
        end
    end
    return (modname=modname, includes=includes, usings=usings)
end

function normalize_include(srcfile::String, inc::String)
    # Includes are relative to the including file's dir
    d = dirname(srcfile)
    p = joinpath(d, inc)
    # If include jumps outside src/, still return normalized path
    return normpath(p)
end

function build_graph()
    nodes = Dict{String,Node}()
    edges = Edge[]
    file_to_mod = Dict{String,String}()

    files = find_src_files()
    for f in files
        parsed = parse_file(f)
        # File node
        nodes[f] = Node(f, f, :file)
        # Module node mapping (optional)
        if parsed.modname !== nothing
            file_to_mod[f] = parsed.modname
            modid = parsed.modname
            nodes[modid] = Node(modid, string(parsed.modname), :module)
        end
        # include edges
        for inc in parsed.includes
            tgt = normalize_include(f, inc)
            nodes[tgt] = haskey(nodes, tgt) ? nodes[tgt] : Node(tgt, tgt, :file)
            push!(edges, Edge(f, tgt, :include))
        end
        # using/import edges
        for mods in parsed.usings
            for m in mods
                # Skip empty or obvious Base
                isempty(m) && continue
                if haskey(nodes, m)
                    push!(edges, Edge(f, m, :using))
                else
                    # External
                    nodes[m] = Node(m, m, :external)
                    push!(edges, Edge(f, m, :using))
                end
            end
        end
    end
    return nodes, edges, file_to_mod
end

function write_mermaid(nodes::Dict{String,Node}, edges::Vector{Edge}; outpath::String)
    open(outpath, "w") do io
        println(io, "graph TD")
        # Group files under a subgraph for readability
        println(io, "  subgraph ParameterCalibration")
        for n in values(nodes)
            n.kind == :file || continue
            @printf(io, "    %s[\"%s\"]\n", replace(n.id, r"[^A-Za-z0-9_]"=>"_"), n.label)
        end
        println(io, "  end")
        # Modules and externals
        for n in values(nodes)
            n.kind != :file || continue
            @printf(io, "  %s[\"%s\"]\n", replace(n.id, r"[^A-Za-z0-9_]"=>"_"), n.label)
        end
        # Edges
        for e in edges
            s = replace(e.src, r"[^A-Za-z0-9_]"=>"_")
            d = replace(e.dst, r"[^A-Za-z0-9_]"=>"_")
            if e.kind == :include
                @printf(io, "  %s --> %s\n", s, d)
            else
                @printf(io, "  %s -. using .-> %s\n", s, d)
            end
        end
    end
end

function write_dot(nodes::Dict{String,Node}, edges::Vector{Edge}; outpath::String)
    open(outpath, "w") do io
        println(io, "digraph G {")
        println(io, "  rankdir=LR;")
        for n in values(nodes)
            shape = n.kind == :file ? "box" : (n.kind == :module ? "ellipse" : "diamond")
            @printf(io, "  \"%s\" [label=\"%s\", shape=%s];\n", n.id, n.label, shape)
        end
        for e in edges
            style = e.kind == :include ? "solid" : "dashed"
            @printf(io, "  \"%s\" -> \"%s\" [style=%s];\n", e.src, e.dst, style)
        end
        println(io, "}")
    end
end

function main()
    nodes, edges, _ = build_graph()
    mkpath("docs/diagrams")
    write_mermaid(nodes, edges; outpath="docs/diagrams/dependencies.mmd")
    write_dot(nodes, edges; outpath="docs/diagrams/dependencies.dot")
    @info "Dependency diagrams written" time=Dates.now() files=(length(nodes), length(edges))
    println("Generated:")
    println("  docs/diagrams/dependencies.mmd")
    println("  docs/diagrams/dependencies.dot")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
