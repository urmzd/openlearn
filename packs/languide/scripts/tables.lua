-- tables.lua: Pandoc Lua filter that sizes table columns from their content.
--
-- Pandoc sizes the columns of a wide pipe table from the dashes in its
-- separator row. A column with a short separator but a long word then
-- overflows into its neighbour. This filter measures the cells instead and
-- gives each column the share of the line that its content needs.

if not FORMAT:match("latex") then
  return {}
end

-- Rough number of average Latin characters that fit on one line.
local LINE_BUDGET = 92
-- Characters lost to the padding of each column.
local COLUMN_PADDING = 2.2

local function char_width(c)
  if c < 0x0300 then return 1 end
  -- Combining marks take no room.
  if (c >= 0x0300 and c <= 0x036F) or (c >= 0x064B and c <= 0x065F)
    or c == 0x0670 or c == 0x200C or c == 0x200D then
    return 0
  end
  -- CJK, kana and full-width forms are about twice as wide.
  if (c >= 0x2E80 and c <= 0xA4CF) or (c >= 0xAC00 and c <= 0xD7AF)
    or (c >= 0xF900 and c <= 0xFAFF) or (c >= 0xFE30 and c <= 0xFE4F)
    or (c >= 0xFF00 and c <= 0xFF60) or (c >= 0x20000 and c <= 0x3FFFF) then
    return 1.9
  end
  -- Cyrillic sets a little wider than Latin.
  if c >= 0x0400 and c <= 0x052F then return 1.1 end
  return 1
end

local function text_width(s)
  local w = 0
  for _, c in utf8.codes(s) do w = w + char_width(c) end
  return w
end

-- Width of the whole cell and of its longest unbreakable piece. CJK text
-- can break between any two characters, so each character is a piece.
local function measure(cell)
  local text = pandoc.utils.stringify(cell.contents)
  local longest = 0
  for word in text:gmatch("[^ \t\r\n]+") do
    local w, piece = 0, 0
    for _, c in utf8.codes(word) do
      local cw = char_width(c)
      if cw > 1.5 then
        piece = cw
        if w > longest then longest = w end
        w = 0
        if piece > longest then longest = piece end
      else
        w = w + cw
      end
    end
    if w > longest then longest = w end
  end
  return text_width(text), longest
end

local function each_row(tbl, fn)
  for _, row in ipairs(tbl.head.rows) do fn(row) end
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.head) do fn(row) end
    for _, row in ipairs(body.body) do fn(row) end
  end
  for _, row in ipairs(tbl.foot.rows) do fn(row) end
end

function Table(tbl)
  local n = #tbl.colspecs
  if n == 0 then return nil end
  local full, word = {}, {}
  for i = 1, n do full[i], word[i] = 0, 0 end
  local spans = false
  each_row(tbl, function(row)
    for i, cell in ipairs(row.cells) do
      if cell.col_span > 1 or cell.row_span > 1 then spans = true end
      if i <= n then
        local f, w = measure(cell)
        if f > full[i] then full[i] = f end
        if w > word[i] then word[i] = w end
      end
    end
  end)
  if spans then return nil end

  local budget = LINE_BUDGET - COLUMN_PADDING * n
  local total = 0
  for i = 1, n do total = total + full[i] end

  -- Everything fits on one line: let LaTeX use the natural widths.
  if total <= budget then
    for i = 1, n do tbl.colspecs[i][2] = nil end
    return tbl
  end

  -- Give narrow columns what they ask for, then share the rest among the
  -- wide columns in proportion to their content.
  local width, settled = {}, {}
  local remaining, open = budget, n
  local changed = true
  while changed and open > 0 do
    changed = false
    local fair = remaining / open
    for i = 1, n do
      if not settled[i] and full[i] <= fair then
        width[i] = full[i]
        settled[i] = true
        remaining = remaining - full[i]
        open = open - 1
        changed = true
      end
    end
  end
  if open > 0 then
    local need = 0
    for i = 1, n do
      if not settled[i] then need = need + math.sqrt(full[i]) end
    end
    for i = 1, n do
      if not settled[i] then
        width[i] = remaining * math.sqrt(full[i]) / need
      end
    end
  end

  -- No column may be narrower than its longest unbreakable piece, up to a
  -- third of the line.
  for i = 1, n do
    local floor = math.min(word[i], budget / 3)
    if width[i] < floor then width[i] = floor end
  end

  local sum = 0
  for i = 1, n do sum = sum + width[i] + COLUMN_PADDING end
  for i = 1, n do
    tbl.colspecs[i][2] = (width[i] + COLUMN_PADDING) / sum
  end
  return tbl
end
