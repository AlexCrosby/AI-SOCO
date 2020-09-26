import regex as re


class Stylometry:
    arithmetic_operators = ['+', '-', '*', '/', '%', '++', '--']
    assignment_operators = ['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '>>=', '<<=']
    comparison_operators = ['==', '!=', '>', '<', '>=', '<=']
    logical_operators = ['&&', '||', '!']
    operators = arithmetic_operators + assignment_operators + comparison_operators + logical_operators
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

    output_dict = {}
    params_dict = {}
    full_code = ''

    def parse(self, code, preprocessed_code):
        self.full_code = code
        output = []
        code_nc, ts1, ts2, fs1 = self.comment_removal(code)
        code_clean = self.quote_removal(code_nc)  # Remove string and char literals so they
        # do not get confused in feature extraction
        # print(code)
        code_nodef = self.quote_removal(preprocessed_code)

        line_count = self.count_lines(code)
        line_count_nc = self.count_lines(code_nc)

        blank_lines = len(re.findall(r"(?<=\n)[ \t]*(?=\n)", code))  # finds and counts empty lines
        operator_count, tl1, ts10 = self.extract_operators(code_nodef)
        tl2 = self.if_statement_layout(code_nodef)
        tl3 = blank_lines / line_count
        tl4 = self.leading_whitespaces(code)
        ts3, ts17, ts18, ts19, ts21 = self.extract_variables(code_clean)
        ts4, ts5 = self.extract_control(code_nodef)
        ts7, ts8, ts9 = self.extract_access(code_clean)
        ts11 = operator_count / line_count_nc
        ts12, ts13, ts14, ts15, ts16 = self.extract_methods(code_clean)
        ts16 /= line_count_nc
        ts20 = ts21 / line_count_nc
        ts22 = self.extract_goto(code_clean)
        tr1 = self.line_length(code)
        vecs = [tl1, tl2, tl3, tl4, ts1, ts2, ts3, ts4, ts5, ts7, ts8, ts9, ts10, ts11, ts12, ts13, ts14, ts15,
                ts16, ts17, ts18, ts19, ts20, ts21, ts22, tr1]
        for v in vecs:
            output.append(v)

        fl2, fl3 = self.for_format(code_nodef)
        fs2 = self.extract_keywords(code_clean)
        fs2 /= line_count_nc
        fs3 = self.loop_count(code_nodef)
        fs3 /= line_count_nc
        # skip fs4
        fs5, fs6 = self.extract_arrays(code_nodef)
        fs7 = self.addition_ops(code_nodef)
        fs8 = 0
        if "return 0;" in code_nodef:
            fs8 = 1
        fs9 = code_nodef.count("#import") / line_count
        # skip fs10
        # fr1 skipped same ts16
        vecs2 = [fl2, fl3, fs1, fs2, fs3, fs5, fs6, fs7, fs8, fs9]
        for v in vecs2:
            output.append(v)
        return output

    def comment_removal(self, code):
        orig_lines = self.count_lines(code)
        code = re.sub(r"(\/\*(.|\n)*?\*\/)\n", '', code)  # Remove multiline comments
        code = re.sub(r"\n\s*//.*", '', code)  # Remove single line comments
        nc_lines = self.count_lines(code)
        hybrid_lines = len(re.findall(r"//.*", code))
        code = re.sub(r"//.*", '', code)  # Remove hybrid line comments
        ts1 = 0
        comment_lines = orig_lines - nc_lines
        if comment_lines < hybrid_lines:
            ts1 = 1
        ts2 = (comment_lines + hybrid_lines) / nc_lines
        fs1 = (comment_lines + hybrid_lines) / orig_lines
        return code, ts1, ts2, fs1

    @staticmethod
    def quote_removal(code):
        # Can't just delete strings as they may be required for a parameter in a #define substitution
        return re.sub(r"[\"\'].*?[\"\']", "\"PLACEHOLDER_STRING\"", code)

    @staticmethod
    def extract_operators(code):
        total_ops = 0
        spaces = 0

        op = re.findall(r"[ \t]*[+\-*\/%=&|^><!]+[ \t]*", code)  # finds any operators
        op = [o for o in op if o != ">>" or o != "<<"]  # >> and << do not count as operators
        unique_ops = [o.strip() for o in op]
        unique_ops = len(set(unique_ops))

        total_ops += len(op)
        for o in op:
            total_len = len(o)  # total length of operator and spaces
            o = o.strip()  # total length of operator alone
            spaces += total_len - len(o)  # length of spaces without operator

        avg_spaces = 0
        if total_ops > 0:
            avg_spaces = spaces / total_ops
        return total_ops, avg_spaces, unique_ops

    @staticmethod
    def if_statement_layout(code):
        same_line = len(re.findall(r"\bif\b[ \t]*\(.*\)[ \t]*{", code))  # if (...) {
        #
        diff_line = len(re.findall(r"\bif\s*\([^\)]*\)[ \t]*\n\s*{", code))  # if (...) \n {
        # Rules
        # Must contain word "if"
        # Then must be followed by parenthesis with anything inside
        # Depending on same line or not, there may be a new line between "{" and parentheses
        # Any spaces or tabs allowed otherwise
        if same_line > diff_line:
            return 1
        else:
            return 0

    @staticmethod
    def leading_whitespaces(code):
        leading_whitespace = 0
        code = code.split('\n')
        total_lines = len(code)
        for line in code:
            full_length = len(line)
            stripped_len = len(line.lstrip())
            leading_whitespace += (full_length - stripped_len)
        return leading_whitespace / total_lines

    def extract_keywords(self, code):
        count = 0
        for keyword in self.keywords:
            count += len(re.findall(r"\b{}\b".format(keyword), code))
        return count

    @staticmethod
    def extract_arrays(code):
        array_2d = min(len(re.findall(r"(?<!\])(\[([^][]*|(?1))*\])[ \t]*(\[([^][]*|(?1))*\])(?!\[)", code)),
                       1)  # finds 2d arrays, return 0 if none, 1 if more than 0
        #   Uses recursion to find outermost [][] pairs so that arrays can have internal array references

        array_formula = [c[0][1:-1] for c in re.findall(r"(\[([^][]*|(?1))*\])", code)]  # 0 gets outer capture group
        array_formula_result = 0
        for c in array_formula:
            if re.match(r".*[^\w\n)(]+.*", c):  # Non-a-Z0-9 and bracket characters indicate some operation
                array_formula_result = 1
                break
        return array_2d, array_formula_result

    def extract_variables(self, code):
        definitions = re.findall(r"#define.+", code)  # Get all #defines
        for i in range(len(definitions)):
            definitions[i] = re.sub("[ \t]*#define[ \t]*", "", definitions[i]).split()[0]  # Get the substitute
        definitions = set(definitions)
        code = re.sub(r"#\b\w+\b.*", '', code)  # Remove all preprocessor directives
        code = re.sub(r"\btypedef\b.*", '', code)  # Remove typedefs
        for keyword in self.keywords:
            code = re.sub(r"\b{}\b".format(keyword), '', code)  # Remove all keywords

        all_vars = re.findall(r"\b[a-zA-Z_][\w]*\b(?!\s*(?:\(|<|>|\"))(?![ \t]*[a-zA-Z_])",  # word
                              code)
        # word  not ending with brackets or  <, > to avoid cout or another wword to avoid definitions and method
        # declarations

        var = set(all_vars)  # Find variables in remainder
        var = var.difference(definitions)  # Exclude #defines from variables
        char_avg = 0
        if len(var) > 0:
            for v in var:
                char_avg += len(v)
            char_avg /= len(var)
        underscores = numbers = uppercase = 0
        char_finder = r"\b\w*[{}]\w*\b"
        for v in var:
            if re.match(char_finder.format("_"), v):
                underscores = 1
            if re.match(char_finder.format("0-9"), v):
                numbers = 1
            if re.match(char_finder.format("A-Z"), v):
                uppercase = 1

        return char_avg, underscores, numbers, uppercase, len(all_vars)

    def extract_methods(self, code):
        types = ['int', 'char', 'void', 'string']
        count = [0, 0, 0, 0]
        code = self.remove_defs(code)
        for keyword in self.keywords:
            code = re.sub(r"\b{}\b".format(keyword), '',
                          code)  # Remove all keywords because for(...) etc are not methods
        methods = re.findall(r"\b[\w]+\b(?=[ \t]+\b[\w]+\b[ \t]*\()", code)  # defined methods
        # word followed by another word then open bracket eg int main (
        for method in methods:
            if method in types:
                count[types.index(method)] += 1
        return count[0], count[1], count[2], count[3], len(methods)

    @staticmethod
    def remove_defs(code):
        return re.sub(r"#\bdefine\b.*", '', code)

    # Returns the number of lines.
    @staticmethod
    def count_lines(code):
        return code.count('\n') + 1

    # Compares occurrences of for:while and if:switch.
    @staticmethod
    def extract_control(code):
        # nodef
        control_statements = ['for', 'while', 'if', 'switch']
        count = [0, 0, 0, 0]
        for cs in range(len(control_statements)):
            count[cs] = len(re.findall(r"\b{}\b".format(control_statements[cs]), code))
        ts4 = 0
        if count[0] < count[1]:
            ts4 = 1
        ts5 = 0
        if count[2] < count[3]:
            ts5 = 0
        return ts4, ts5

        # Undoes a users #define directives by substituting the full code back in. Also removes #define lines following
        # this.

    def undo_defs(self, code):
        code = re.sub("[ \t]+[\\\\][\n][ \t]*", " ", code)  # Forces multiline #defs onto single line
        definitions = re.findall(r"#define.+", code)  # String for each #define line
        definitions_dict = {}
        self.output_dict = {}
        self.params_dict = {}
        code = re.sub(r"#define.+", "", code)

        for i in range(len(definitions)):
            definitions[i] = re.sub("[ \t]*#define[ \t]*", "", definitions[i])
            split = re.match(r"(?<first>\w+(?:\(.*?\))?)(?:[ \t]*)(?<second>.*)",
                             definitions[i])  # matched definition lines
            try:
                definitions[i] = [split.group('first'), split.group('second')]
            except:
                # If an exception occurs here, a #define does not conform to expected style.
                print(definitions)
                print(definitions[i])
                print(self.full_code)
                exit()

            func_name = definitions[i][0]
            definitions_dict[func_name] = definitions[i][1]

        # print(definitions_dict)
        list_of_keys = list(definitions_dict)

        for item in list_of_keys:
            try:
                if "(" not in item:
                    code = re.sub(r"(?<![\w]){}(?![\w])".format(item), definitions_dict[item], code)
            except Exception as e:
                print(self.full_code)
                print(e)
                print(item)
                print(definitions_dict[item])
                exit(-2)
            if "(" in item:
                # print(item)
                if '...' in item:
                    continue
                split = item.split('(', maxsplit=1)
                name = split[0]
                params = split[1][:-1].split(',')
                self.output_dict[name] = definitions_dict[item]
                self.params_dict[name] = params
                code = re.sub(r"(?<!\w)(?<a>{})[ \t]*(?<b>\((?:[^)(]*|(?1))*\))".format(name), self.rewrite_method,
                              code)
        return code

    # Used by undo_defs to rebuild #define functions with new parameters.
    def rewrite_method(self, n):
        a = n.group('a')
        b = n.group('b')[1:-1].split(',')  # sub in
        b = [x.strip() for x in b]



        output = self.output_dict[a]
        for i in range(len(self.params_dict[a])):
            output = re.sub(r"\b{}\b".format(self.params_dict[a][i]), b[i], output)

        return output

    @staticmethod
    def extract_access(code): # Gets relative count of 'public', 'private' and 'protected.
        access_list = ['public', 'private', 'protected']
        count = [0, 0, 0]
        for i in range(len(access_list)):
            count[i] = len(re.findall("\b{}\b".format(access_list[i]), code))
        total = count[0] + count[1] + count[2]
        if total == 0:
            return 0, 0, 0
        else:
            return count[0] / total, count[1] / total, count[2] / total

    @staticmethod
    def extract_goto(code): # Finds usage of 'go to'
        return min(len(re.findall(r"\bgo\b[ \t]+\bto\b", code)), 1)

    def line_length(self, code): # Gets average characters per line.
        return len(code) / self.count_lines(code)

    #####################################################################
    @staticmethod
    def for_format(code):
        space_presence = 0  # Spaces prese
        space_total = 0
        contents = re.findall(r"(?:\bfor\b[ \t]*)\((.+)\)", code)
        if len(contents) == 0:
            return 0, 0

        for c in contents:
            space_total += c.count(' ')
        if space_total > 0:
            space_presence = 1
        space_total /= len(contents)
        return space_total, space_presence

    @staticmethod
    def loop_count(code):
        loops = 0
        loops += len(re.findall(r"\bfor\b", code))
        loops += len(re.findall(r"\bwhile\b", code))
        return loops

    @staticmethod
    def addition_ops(code): # Finds preferred addition structure.
        short = len(re.findall(r"(?<![+\-*\/%=&|^><!])\+=(?![+\-*\/%=&|^><!])", code))  # eg n+=1
        normal = len(re.findall(r"\b\w+\b[ \t]*=[ \t]*\b\w+\b[ \t]*\+[ \t]*\b\w+\b", code))  # eg n=n+1
        if short == 0:
            return 0
        return short / (short + normal)
