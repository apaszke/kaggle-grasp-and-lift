PRINT = print

function print(str)
    PRINT(os.date('[%H:%M:%S] ') .. tostring(str))
end

function printRed(str)
    print('\27[0;31m' .. tostring(str) .. '\27[m')
end

function printGreen(str)
    print('\27[0;32m' .. tostring(str) .. '\27[m')
end

function printYellow(str)
    print('\27[0;33m' .. tostring(str) .. '\27[m')
end

function printBlue(str)
    print('\27[0;34m' .. tostring(str) .. '\27[m')
end
