# rag-methods-playground
Description: A playground for learning and experimenting with Retrieval-Augmented Generation (RAG) methods, including dense and hybrid retrieval, agent-based orchestration, multi-document synthesis, semantic search, evaluation frameworks, and production best practices.





## Reinstall Virtual environment

(.venv) @btholath ➜ /workspaces/rag-methods-playground (main) $ cat /etc/os-release
NAME="Ubuntu"
VERSION="20.04.6 LTS (Focal Fossa)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 20.04.6 LTS"
VERSION_ID="20.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=focal
UBUNTU_CODENAME=focal
(.venv) @btholath ➜ /workspaces/rag-methods-playground (main) $ 

sudo apt update
sudo apt install sqlite3
(.venv) @btholath ➜ /workspaces/rag-methods-playground (main) $ sudo apt install sqlite3
Reading package lists... Done
Building dependency tree       
Reading state information... Done
sqlite3 is already the newest version (3.31.1-4ubuntu0.7).
0 upgraded, 0 newly installed, 0 to remove and 82 not upgraded.
(.venv) @btholath ➜ /workspaces/rag-methods-playground (main) $ 

(.venv) @btholath ➜ /workspaces/rag-methods-playground (main) $ sqlite3 --version
3.45.3 2024-04-15 13:34:05 8653b758870e6ef0c98d46b3ace27849054af85da891eb121e9aaa537f1e8355 (64-bit)

(.venv) @btholath ➜ /workspaces/rag-methods-playground (main) $ 


# Optional: Remove old venv
rm -rf .venv
# Re-create venv with correct python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

@btholath ➜ /workspaces/rag-methods-playground (main) $ sudo apt-get install -y sqlite3 libsqlite3-dev
Reading package lists... Done
Building dependency tree       
Reading state information... Done
libsqlite3-dev is already the newest version (3.31.1-4ubuntu0.7).
sqlite3 is already the newest version (3.31.1-4ubuntu0.7).
0 upgraded, 0 newly installed, 0 to remove and 82 not upgraded.

@btholath ➜ /workspaces/rag-methods-playground (main) $ sqlite3 --version   # Should show 3.35.0 or higher
3.45.3 2024-04-15 13:34:05 8653b758870e6ef0c98d46b3ace27849054af85da891eb121e9aaa537f1e8355 (64-bit)

@btholath ➜ /workspaces/rag-methods-playground (main) $ curl https://pyenv.run | bash
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   270  100   270    0     0   1560      0 --:--:-- --:--:-- --:--:--  1560

WARNING: Can not proceed with installation. Kindly remove the '/home/codespace/.pyenv' directory first.

@btholath ➜ /workspaces/rag-methods-playground (main) $ rm -rf /home/codespace/.pyenv

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

@btholath ➜ /workspaces/rag-methods-playground (main) $ source ~/.bashrc

@btholath ➜ /workspaces/rag-methods-playground (main) $ pyenv --version
pyenv 2.6.1

pyenv install 3.12.1
pyenv global 3.12.1

Python’s SQLite version
import sqlite3
print(sqlite3.sqlite_version)


(.venv) @btholath ➜ /workspaces/rag-methods-playground/reranking_cross_encoder (main) $ python reranking.py 
Traceback (most recent call last):
  File "/workspaces/rag-methods-playground/reranking_cross_encoder/reranking.py", line 9, in <module>
    import chromadb
  File "/workspaces/rag-methods-playground/.venv/lib/python3.12/site-packages/chromadb/__init__.py", line 94, in <module>
    raise RuntimeError(
RuntimeError: Your system has an unsupported version of sqlite3. Chroma                     requires sqlite3 >= 3.35.0.
Please visit                     https://docs.trychroma.com/troubleshooting#sqlite to learn how                     to upgrade.
(.venv) @btholath ➜ /workspaces/rag-methods-playground/reranking_cross_encoder (main) $ 

Since Python is linked to its internal SQLite, you need to explicitly tell Python to use the system-installed SQLite (3.45.3).


(.venv) @btholath ➜ /workspaces/rag-methods-playground/reranking_cross_encoder (main) $ python
Python 3.12.1 (main, Mar 17 2025, 17:13:06) [GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import sqlite3
>>> print(sqlite3.__file__)
/home/codespace/.pyenv/versions/3.12.1/lib/python3.12/sqlite3/__init__.py
>>> print(sqlite3.sqlite_version)
3.31.1
>>> 

sudo apt install libsqlite3-dev
pip uninstall pysqlite3-binary
pip install pysqlite3-binary
import sqlite3
print(sqlite3.sqlite_version)
pip uninstall chromadb
pip install chromadb

sudo apt remove libsqlite3-dev
sudo apt autoremove
wget https://www.sqlite.org/2024/sqlite-autoconf-3450300.tar.gz
tar xvf sqlite-autoconf-3450300.tar.gz
cd sqlite-autoconf-3450300
./configure
make
sudo make install
sqlite3 --version
pip uninstall pysqlite3-binary
pip install pysqlite3-binary
pip uninstall chromadb
pip install chromadb
import sqlite3
print(sqlite3.sqlite_version)

pip install 'chromadb==0.3.22'

