-- rtl.lua: Pandoc Lua filter that typesets right-to-left script (Arabic,
-- Persian) correctly under XeLaTeX.
--
-- XeTeX shapes each Arabic word on its own, but it lays words out left to
-- right. This filter finds runs of right-to-left words and wraps them in
-- \RLtext{...}, which the template defines with TeX--XeT. Inside a run it
-- keeps numbers left to right and mirrors brackets, as the Unicode
-- bidirectional algorithm does.

if not FORMAT:match("latex") then
  return {}
end

local function is_rtl_cp(c)
  return (c >= 0x0590 and c <= 0x08FF)
    or (c >= 0xFB1D and c <= 0xFDFF)
    or (c >= 0xFE70 and c <= 0xFEFF)
end

local function is_rtl_digit(c)
  return (c >= 0x0660 and c <= 0x0669) or (c >= 0x06F0 and c <= 0x06F9)
end

-- Classify a string: "rtl" has a right-to-left letter, "ltr" has any other
-- letter, "num" has only digits and punctuation, "neutral" has only
-- punctuation.
local function classify(s)
  local rtl, ltr, digit = false, false, false
  for _, c in utf8.codes(s) do
    if is_rtl_digit(c) or (c >= 0x30 and c <= 0x39) then
      digit = true
    elseif is_rtl_cp(c) then
      rtl = true
    elseif (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c >= 0x00C0 then
      -- Letters outside ASCII punctuation. General punctuation stays neutral.
      if not (c >= 0x2000 and c <= 0x206F) then
        ltr = true
      end
    end
  end
  if rtl then return "rtl" end
  if ltr then return "ltr" end
  if digit then return "num" end
  return "neutral"
end

local escapes = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}",
  ["#"] = "\\#", ["$"] = "\\$", ["%"] = "\\%", ["&"] = "\\&",
  ["_"] = "\\_", ["^"] = "\\^{}", ["~"] = "\\textasciitilde{}",
}

local mirror = { ["("] = ")", [")"] = "(", ["["] = "]", ["]"] = "[" }

local function escape(s)
  local out = {}
  for _, c in utf8.codes(s) do
    local ch = utf8.char(c)
    out[#out + 1] = escapes[ch] or ch
  end
  return table.concat(out)
end

-- Character kinds inside a right-to-left run. The Arabic-script font has no
-- ASCII glyphs, so ASCII digits and punctuation switch back to the text font.
local function char_kind(c)
  if c >= 0x30 and c <= 0x39 then return "latin_num" end
  if is_rtl_digit(c) then return "rtl_num" end
  if c >= 0x21 and c <= 0x7E then return "punct" end
  return "rtl"
end

-- Characters that may sit inside a number, such as 10:30 or 1,500.
local function joins_number(c)
  return c == 0x2E or c == 0x2C or c == 0x3A or c == 0x2F or c == 0x25
    or c == 0x066B or c == 0x066C
end

local function emit(kind, text)
  if kind == "latin_num" then
    return "\\LRnum{" .. escape(text) .. "}"
  elseif kind == "rtl_num" then
    return "\\RLdigits{" .. escape(text) .. "}"
  elseif kind == "punct" then
    local out = {}
    for _, c in utf8.codes(text) do
      local ch = utf8.char(c)
      out[#out + 1] = escapes[mirror[ch] or ch] or mirror[ch] or ch
    end
    return "{\\normalfont " .. table.concat(out) .. "}"
  end
  return escape(text)
end

-- Render one word of a right-to-left run. Numbers stay left to right and
-- brackets are mirrored.
local function rtl_word(s)
  local cps = {}
  for _, c in utf8.codes(s) do cps[#cps + 1] = c end
  local out, buf, buf_kind = {}, {}, nil
  local function flush()
    if #buf > 0 then
      out[#out + 1] = emit(buf_kind, table.concat(buf))
      buf = {}
    end
  end
  for idx, c in ipairs(cps) do
    local kind = char_kind(c)
    -- A separator between two digits belongs to the number.
    if (buf_kind == "latin_num" or buf_kind == "rtl_num") and joins_number(c)
      and cps[idx + 1] and char_kind(cps[idx + 1]) == buf_kind then
      kind = buf_kind
    end
    if kind ~= buf_kind then flush() end
    buf_kind = kind
    buf[#buf + 1] = utf8.char(c)
  end
  flush()
  return table.concat(out)
end

local function render_run(items)
  local out = {}
  for _, el in ipairs(items) do
    if el.t == "Space" or el.t == "SoftBreak" then
      out[#out + 1] = " "
    else
      out[#out + 1] = rtl_word(el.text)
    end
  end
  return table.concat(out)
end

local function has_rtl_digit(s)
  for _, c in utf8.codes(s) do
    if is_rtl_digit(c) then return true end
  end
  return false
end

local function wrap_inlines(inlines)
  local result = pandoc.List()
  local i, n = 1, #inlines
  while i <= n do
    local el = inlines[i]
    if el.t == "Str" and classify(el.text) == "rtl" then
      -- Extend the run over spaces, numbers and punctuation, as long as
      -- another right-to-left word follows.
      local last = i
      local j = i + 1
      while j <= n do
        local nx = inlines[j]
        if nx.t == "Space" or nx.t == "SoftBreak" then
          j = j + 1
        elseif nx.t == "Str" then
          local kind = classify(nx.text)
          if kind == "rtl" then
            last = j
            j = j + 1
          elseif kind == "ltr" then
            break
          else
            j = j + 1
          end
        else
          break
        end
      end
      local run = {}
      for k = i, last do run[#run + 1] = inlines[k] end
      result:insert(pandoc.RawInline("latex", "\\RLtext{" .. render_run(run) .. "}"))
      i = last + 1
    elseif el.t == "Str" and has_rtl_digit(el.text) then
      -- A lone number in Arabic-script digits still needs the Arabic font.
      local parts, digits = {}, {}
      local function flush_digits()
        if #digits > 0 then
          parts[#parts + 1] = "\\RLdigits{" .. table.concat(digits) .. "}"
          digits = {}
        end
      end
      for _, c in utf8.codes(el.text) do
        if is_rtl_digit(c) or c == 0x066B or c == 0x066C then
          digits[#digits + 1] = utf8.char(c)
        else
          flush_digits()
          parts[#parts + 1] = escape(utf8.char(c))
        end
      end
      flush_digits()
      result:insert(pandoc.RawInline("latex", table.concat(parts)))
      i = i + 1
    else
      result:insert(el)
      i = i + 1
    end
  end
  return result
end

local function all_rtl(inlines)
  local seen = false
  for _, el in ipairs(inlines) do
    if el.t == "Str" then
      local kind = classify(el.text)
      if kind == "ltr" then return false end
      if kind == "rtl" then seen = true end
    elseif el.t ~= "Space" and el.t ~= "SoftBreak" and el.t ~= "LineBreak" then
      return false
    end
  end
  return seen
end

-- A paragraph that holds only right-to-left text is set flush right.
local function para(el)
  if all_rtl(el.content) then
    local blocks = pandoc.List()
    blocks:insert(pandoc.RawBlock("latex", "\\begin{RLpara}"))
    blocks:insert(pandoc.Para(wrap_inlines(el.content)))
    blocks:insert(pandoc.RawBlock("latex", "\\end{RLpara}"))
    return blocks
  end
end

-- Inline code cannot carry right-to-left text in a monospaced font, so
-- such code becomes ordinary text.
local function code(el)
  if classify(el.text) == "rtl" then
    local parts = pandoc.List()
    local first = true
    for word in el.text:gmatch("[^ \t\r\n]+") do
      if not first then parts:insert(pandoc.Space()) end
      parts:insert(pandoc.Str(word))
      first = false
    end
    return wrap_inlines(parts)
  end
end

local function code_block(el)
  if classify(el.text) == "rtl" then
    local lines = pandoc.List()
    for line in (el.text .. "\n"):gmatch("(.-)\n") do
      local parts = pandoc.List()
      local first = true
      for word in line:gmatch("[^ \t\r\n]+") do
        if not first then parts:insert(pandoc.Space()) end
        parts:insert(pandoc.Str(word))
        first = false
      end
      lines:insert(parts)
    end
    return pandoc.LineBlock(lines)
  end
end

return {
  { Code = code, CodeBlock = code_block },
  { Para = para },
  { Inlines = wrap_inlines },
}
