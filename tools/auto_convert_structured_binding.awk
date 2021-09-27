function convert(indent, line) {
  match(line, /^(.*auto\s+&*)\[([^\]]*)\](.*)$/, arr) # certainly always match
  split(arr[2], vars, /\s*,\s*/)
  mixvar = "_"
  for (vi in vars) {
    mixvar = mixvar vars[vi] "_"
  }
  print arr[1] mixvar arr[3]
  for (vi in vars) {
    print indent "auto &" vars[vi] " = std::get<" (vi-1) ">(" mixvar ");"
  }
}

match($0, /^(\s+)auto\s+&*\[/, arr) {
  convert(arr[1], $0)
  next
}

match($0, /^(\s+)for.*auto\s+&*\[/, arr) {
  convert(arr[1]"  ", $0)
  next
}

{
  print
}
