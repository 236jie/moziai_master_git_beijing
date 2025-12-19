# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['tune_evaluate.py'],
             pathex=['C:\\Users\\Administrator\\Desktop\\nlz\\mozi_ai_sdk\\nlz_wrj\\bin'],
             binaries=[],
             datas=[],
             hiddenimports=['ray.external_storage','ray.async_compat','msgpack','scipy.special.cython_special'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [("checkpoint-99","C:\\Users\\Administrator\\Desktop\\nlz\\mozi_ai_sdk\\nlz_wrj\\checkpoint\\checkpoint_99\\checkpoint-99","DATA"),],
          name='tune_evaluate',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
