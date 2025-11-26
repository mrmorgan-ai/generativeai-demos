# Serverless Computing
Examples are made using azure cloud resources
## Installing Azure Functions Core Tools
### Windows
1. **Using Windows Package Manager (winget)**
   ```
   winget install Microsoft.Azure.FunctionsCoreTools
   ```
2. **Alternative: Using Chocolatey**
   ```
   choco install azure-functions-core-tools
   ```
3. **Alternative: Using npm**
   ```
   npm install -g azure-functions-core-tools@4 --unsafe-perm true
   ```
4. **Verify installation**
   ```
   func --version
   ```
## Linux
1. **Ubuntu/Debian - Add Microsoft package repository**
   ```bash
   curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
   sudo mv microsoft.gpg /etc/apt/trusted.gpg.d/microsoft.gpg
   ```
2. **Add the package source**
   ```bash
   sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/microsoft-ubuntu-$(lsb_release -cs)-prod $(lsb_release -cs) main" > /etc/apt/sources.list.d/dotnetdev.list'
   ```
3. **Install the tools**
   ```bash
   sudo apt-get update
   sudo apt-get install azure-functions-core-tools-4
   ```
4. **Alternative: Using npm (any distro)**
   ```bash
   npm install -g azure-functions-core-tools@4 --unsafe-perm true
   ```
## macOS

1. **Using Homebrew**
   ```bash
   brew tap azure/functions
   brew install azure-functions-core-tools@4
   ```
2. **Alternative: Using npm**
   ```bash
   npm install -g azure-functions-core-tools@4 --unsafe-perm true
   ```

3. **Verify installation**
   ```bash
   func --version
   ```
**Note:** The npm method works across all platforms but requires Node.js to be installed first.
