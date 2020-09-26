import regex as re
# Example technique to unfold #define preprocessor directives.
# This was deprecated in favour of astminer later in development.
keywords = ['alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
            'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'constexpr', 'const_cast',
            'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit',
            'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable',
            'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'private',
            'protected', 'public', 'register', 'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static',
            'static_assert', 'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true',
            'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
            'wchar_t', 'while', 'xor', 'xor_eq', 'int8_t', 'uint8_t', 'int16_t', 'uint16_t', 'int32_t', 'uint32_t',
            'int64_t', 'uint64_t']

code = r'''#include <bits/stdc++.h>
using namespace std;

#define fastIO                   \
    ios::sync_with_stdio(false); \
    cin.tie(NULL);               \
    cout.tie(NULL);
#define asc(s) sort(s.begin(), s.end())
#define des(s) sort(s.rbegin(), s.rend())
#define pb(x) push_back(x)
#define mp(x, y) make_pair(x, y)
#define all(v) v.begin(), v.end()
#define rev(v) reverse(v.begin(), v.end())
#define lower(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define upper(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define precision(x, p) fixed << setprecision(p) << x
#define set_bits(n) __builtin_popcount(n);
#define MOD 1000000007
#define PI 3.14159265358979



int main()
{
    fastIO
}
'''


def spec(n):
    a = n.group('a')
    b = n.group('b')[1:-1].split(',')  # sub in
    b = [x.strip() for x in b]
    print(a)
    print(b)

    print(params_dict[a])  # asts_old
    output = output_dict[a]
    for i in range(len(params_dict[a])):
        output = re.sub(r"\b{}\b".format(params_dict[a][i]), b[i], output)
    print(output_dict[a])
    print(output)
    return output

code = re.sub("[ \t]+[\\\\][\n][ \t]*"," ",code)
definitions = re.findall(r"#define.+", code)  # String for each #define line
definitions_dict = {}
output_dict = {}
params_dict = {}
code = re.sub(r"#define.+", "", code)

for i in range(len(definitions)):
    definitions[i] = re.sub("[ \t]*#define[ \t]*", "", definitions[i])
    split = re.match(r"(?<first>\w+(?:\(.*?\))?)(?:[ \t]*)(?<second>.*)", definitions[i])
    definitions[i] = [split.group('first'), split.group('second')]

    func_name = definitions[i][0]
    definitions_dict[func_name] = definitions[i][1]

print(definitions_dict)
list_of_keys = list(definitions_dict)

for item in list_of_keys:
    if "(" not in item:
        code = re.sub(r"(?<![\w]){}(?![\w])".format(item), definitions_dict[item], code)
    if "(" in item:
        split = item.split('(',maxsplit=1)
        name = split[0]
        params = split[1][:-1].split(',')
        output_dict[name] = definitions_dict[item]
        params_dict[name] = params
        # print("START")
        code = re.sub(r"(?<!\w)(?<a>{})[ \t]*(?<b>\((?:[^)(]*|(?1))*\))".format(name), spec, code)

print(code)
