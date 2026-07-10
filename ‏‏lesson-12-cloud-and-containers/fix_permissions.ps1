# 1. Store the path to your key
$path = "C:\Eran\aws_test_key_1.pem"

# 2. Reset the permissions to remove inheritance
icacls $path /c /t /inheritance:d

# 3. Remove permissions for "Authenticated Users" (the one mentioned in your error)
icacls $path /c /t /remove "NT AUTHORITY\Authenticated Users"

# 4. Remove permissions for "Users"
icacls $path /c /t /remove "BUILTIN\Users"

# 5. Ensure ONLY you have read access (re-grant to current user)
icacls $path /c /t /grant:r "${env:USERNAME}:(R)"