"""Pure parser regressions; no live HPC access or job submission."""
from pathlib import Path
import os,shutil,subprocess,unittest

HERE=Path(__file__).resolve().parent
CANDIDATES=[HERE/'parse_dicc_account_status.sh', HERE.parent/'tools/parse_dicc_account_status.sh']
PARSER=next(p for p in CANDIDATES if p.is_file())
if os.name=='nt':
    BASH=Path(os.environ.get('PROGRAMFILES','C:/Program Files'))/'Git/bin/bash.exe'
else:BASH=Path(shutil.which('bash') or '/nonexistent/bash')

class AccountStatusTests(unittest.TestCase):
    def check(self,source,expected=None):
        r=subprocess.run([str(BASH),str(PARSER)],input=source.encode(),capture_output=True,timeout=10)
        if expected is None:
            self.assertEqual(r.returncode,77,r.stderr.decode())
            self.assertEqual(r.stdout,b'')
        else:
            self.assertEqual(r.returncode,0,r.stderr.decode())
            self.assertEqual(r.stdout.decode().strip(),expected)
    def test_plain_full(self):self.check(' Current Status : FULL\n','full')
    def test_plain_limited(self):self.check(' Current Status : LIMITED\n','limited')
    def test_observed_colored_full(self):
        self.check('\x1b[2m\x1b[37m Account Information [Updated Hourly]\x1b[0m\n'
                   ' Current Status       : \x1b[36mFULL\x1b[0m\n'
                   ' Job Resource Limits:\n  Maximum Walltime : \x1b[36m3 days\x1b[0m\n','full')
    def test_colored_limited_crlf(self):self.check('Current Status: \x1b[1;36mLIMITED\x1b[0m\r\n','limited')
    def test_colored_field_label(self):self.check('\x1b[36mCurrent Status\x1b[0m:\x1b[32mFULL\x1b[0m\n','full')
    def test_whitespace_and_case(self):self.check('\tcurrent status : full  \n','full')
    def test_inactive_values(self):
        for v in ['SUSPENDED','DISABLED','EXPIRED','INACTIVE']:
            with self.subTest(value=v):self.check('Current Status: \x1b[31m'+v+'\x1b[0m\n')
    def test_empty(self):self.check('')
    def test_unknown(self):self.check('Current Status: PENDING\n')
    def test_unrelated_full_word(self):self.check('Contact support for FULL access\n')
    def test_duplicate_fields(self):self.check('Current Status: FULL\nCurrent Status: FULL\n')
    def test_conflicting_fields(self):self.check('Current Status: FULL\nCurrent Status: LIMITED\n')
    def test_annotated_status_rejected(self):self.check('Current Status: FULL (pending)\n')
    def test_shell_text_not_executed(self):self.check('Current Status: FULL; printf injected\n')
    def test_inactive_notice_fails_closed(self):self.check('Current Status: FULL\nAccount suspended\n')
    def test_old_boundary_pattern_reproduces_failure(self):
        r=subprocess.run([str(BASH),'-c',"grep -Eqi '\\bFULL\\b'"],
                         input=b'Current Status : \x1b[36mFULL\x1b[0m\n',capture_output=True,timeout=10)
        self.assertEqual(r.returncode,1)

if __name__=='__main__':unittest.main()
