--[[ Minimaler JSON-Decoder (Objekte, Arrays, Strings, Zahlen, true/false/null). ]]

local JSON = {}

local function skip(s, i)
  local _, e = s:find("^[ \n\r\t]*", i)
  return e + 1
end

local decode_value

local function decode_string(s, i)
  local out, j = {}, i + 1
  while true do
    local c = s:sub(j, j)
    if c == "" then error("JSON: Zeichenkette nicht beendet") end
    if c == '"' then return table.concat(out), j + 1 end
    if c == "\\" then
      local n = s:sub(j + 1, j + 1)
      local map = { b = "\b", f = "\f", n = "\n", r = "\r", t = "\t", ['"'] = '"', ["\\"] = "\\", ["/"] = "/" }
      if n == "u" then
        local code = tonumber(s:sub(j + 2, j + 5), 16)
        if code < 0x80 then
          out[#out + 1] = string.char(code)
        elseif code < 0x800 then
          out[#out + 1] = string.char(0xC0 + math.floor(code / 0x40), 0x80 + code % 0x40)
        else
          out[#out + 1] = string.char(0xE0 + math.floor(code / 0x1000), 0x80 + math.floor(code / 0x40) % 0x40,
            0x80 + code % 0x40)
        end
        j = j + 6
      else
        out[#out + 1] = map[n] or n
        j = j + 2
      end
    else
      out[#out + 1] = c
      j = j + 1
    end
  end
end

local function decode_array(s, i)
  local arr, j = {}, skip(s, i + 1)
  if s:sub(j, j) == "]" then return arr, j + 1 end
  while true do
    local v
    v, j = decode_value(s, j)
    arr[#arr + 1] = v
    j = skip(s, j)
    local c = s:sub(j, j)
    if c == "]" then return arr, j + 1 end
    if c ~= "," then error("JSON: ',' erwartet in Liste") end
    j = skip(s, j + 1)
  end
end

local function decode_object(s, i)
  local obj, j = {}, skip(s, i + 1)
  if s:sub(j, j) == "}" then return obj, j + 1 end
  while true do
    local k
    k, j = decode_string(s, j)
    j = skip(s, j)
    if s:sub(j, j) ~= ":" then error("JSON: ':' erwartet") end
    local v
    v, j = decode_value(s, skip(s, j + 1))
    obj[k] = v
    j = skip(s, j)
    local c = s:sub(j, j)
    if c == "}" then return obj, j + 1 end
    if c ~= "," then error("JSON: ',' erwartet in Objekt") end
    j = skip(s, j + 1)
  end
end

decode_value = function(s, i)
  i = skip(s, i)
  local c = s:sub(i, i)
  if c == "{" then return decode_object(s, i) end
  if c == "[" then return decode_array(s, i) end
  if c == '"' then return decode_string(s, i) end
  if s:sub(i, i + 3) == "true" then return true, i + 4 end
  if s:sub(i, i + 4) == "false" then return false, i + 5 end
  if s:sub(i, i + 3) == "null" then return nil, i + 4 end
  local num = s:match("^-?%d+%.?%d*[eE]?[-+]?%d*", i)
  if num and #num > 0 then return tonumber(num), i + #num end
  error("JSON: unerwartetes Zeichen '" .. c .. "' an Position " .. i)
end

function JSON.decode(s)
  local v = decode_value(s, 1)
  return v
end

return JSON
