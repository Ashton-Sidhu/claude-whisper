# Homebrew Distribution Plan for claude-whisper

## Overview

This plan outlines the complete strategy for distributing claude-whisper through Homebrew using a custom tap. The package is a Python CLI tool with native dependencies (portaudio) and requires a specialized installation approach.

## Distribution Strategy

**Approach:** Create a custom Homebrew tap (`homebrew-claude-whisper`)

**Rationale:**
- Full control over versioning and updates
- Faster iteration without core maintainer review
- Better for specialized/niche tools
- Can migrate to homebrew-core later once stable

## Prerequisites

### Build System Compatibility Issue

**Current:** Package uses `uv_build` backend (pyproject.toml:24-26)
**Problem:** Homebrew doesn't support uv_build out of the box
**Solution:** Add alternative build backend for Homebrew compatibility

The pyproject.toml needs to support standard Python build tools:

```toml
[build-system]
requires = ["hatchling>=1.0.0"]
build-backend = "hatchling.build"
```

This maintains dev workflow with uv while enabling Homebrew distribution.

**Critical File:** `/Users/sidhu/claude-whisper/pyproject.toml`

## Implementation Steps

### Phase 1: Create Homebrew Tap Repository

**Step 1.1: Create Repository**
```bash
gh repo create homebrew-claude-whisper --public \
  --description "Homebrew formula for claude-whisper voice interface"
```

**Step 1.2: Setup Repository Structure**
```bash
git clone git@github.com:Ashton-Sidhu/homebrew-claude-whisper.git
cd homebrew-claude-whisper
mkdir -p Formula .github/workflows
```

**Step 1.3: Create Tap README**

Create `README.md`:
```markdown
# Homebrew Tap for claude-whisper

Voice-controlled interface for Claude Code using Apple MLX Whisper.

## Installation

```bash
brew tap Ashton-Sidhu/claude-whisper
brew install claude-whisper
```

## Requirements

- macOS with Apple Silicon
- Anthropic API key

## Usage

```bash
export ANTHROPIC_API_KEY="your-key"
claude-whisper /path/to/project
```

See [main repository](https://github.com/Ashton-Sidhu/claude-whisper) for full documentation.
```

### Phase 2: Generate Formula Resources

**Step 2.1: Install homebrew-pypi-poet**
```bash
brew install homebrew-pypi-poet
```

**Step 2.2: Generate Resource Stanzas**
```bash
cd /Users/sidhu/claude-whisper
poet -f claude-whisper > /tmp/resources.rb
```

This generates all Python dependency resource blocks with URLs and SHA256 hashes.

**Step 2.3: Review Generated Resources**

The resources file will contain stanzas for all dependencies:
- claude-agent-sdk
- desktop-notifier
- loguru
- mlx-whisper
- pyaudio
- pydantic
- pydantic-settings
- pynput
- All transitive dependencies (annotated-types, pydantic-core, mlx, numpy, etc.)

### Phase 3: Create Homebrew Formula

**Step 3.1: Create Formula File**

Create `Formula/claude-whisper.rb`:

```ruby
class ClaudeWhisper < Formula
  include Language::Python::Virtualenv

  desc "Voice-controlled interface for Claude Code using MLX Whisper"
  homepage "https://github.com/Ashton-Sidhu/claude-whisper"
  url "https://github.com/Ashton-Sidhu/claude-whisper/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "GENERATED_SHA256_HASH"
  license "BSD-3-Clause"

  depends_on "portaudio"
  depends_on "python@3.13"

  # Resource stanzas from poet output
  resource "claude-agent-sdk" do
    url "https://files.pythonhosted.org/packages/.../claude-agent-sdk-0.1.18.tar.gz"
    sha256 "..."
  end

  resource "desktop-notifier" do
    url "https://files.pythonhosted.org/packages/.../desktop-notifier-6.2.0.tar.gz"
    sha256 "..."
  end

  resource "loguru" do
    url "https://files.pythonhosted.org/packages/.../loguru-0.7.3.tar.gz"
    sha256 "..."
  end

  resource "mlx-whisper" do
    url "https://files.pythonhosted.org/packages/.../mlx_whisper-0.4.3.tar.gz"
    sha256 "..."
  end

  resource "pyaudio" do
    url "https://files.pythonhosted.org/packages/.../PyAudio-0.2.14.tar.gz"
    sha256 "..."
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/.../pydantic-2.10.5.tar.gz"
    sha256 "..."
  end

  resource "pydantic-settings" do
    url "https://files.pythonhosted.org/packages/.../pydantic_settings-2.7.1.tar.gz"
    sha256 "..."
  end

  resource "pynput" do
    url "https://files.pythonhosted.org/packages/.../pynput-1.8.1.tar.gz"
    sha256 "..."
  end

  # Add all transitive dependencies from poet output

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      claude-whisper requires macOS permissions:

      System Settings > Privacy & Security:
        • Input Monitoring (for push-to-talk)
        • Accessibility (for keyboard monitoring)

      Configuration: ~/.config/claude-whisper/config.toml

      Set your API key:
        export ANTHROPIC_API_KEY="your-key"
    EOS
  end

  test do
    assert_predicate bin/"claude-whisper", :exist?
    output = shell_output("#{bin}/claude-whisper --help 2>&1", 2)
    assert_match "usage: claude-whisper", output
    system libexec/"bin/python", "-c", "import claude_whisper"
  end
end
```

**Step 3.2: Generate Source Tarball Hash**
```bash
curl -sL https://github.com/Ashton-Sidhu/claude-whisper/archive/refs/tags/v0.1.0.tar.gz | sha256sum
```

Replace `GENERATED_SHA256_HASH` in formula with actual hash.

**Step 3.3: Insert Resource Stanzas**

Copy all resource blocks from `/tmp/resources.rb` into the formula file.

### Phase 4: Test Formula Locally

**Step 4.1: Test Installation**
```bash
cd homebrew-claude-whisper
brew install --build-from-source ./Formula/claude-whisper.rb
```

**Step 4.2: Run Formula Tests**
```bash
brew test claude-whisper
```

**Step 4.3: Audit Formula**
```bash
brew audit --strict claude-whisper
```

**Step 4.4: Verify Command Works**
```bash
claude-whisper --help
```

**Step 4.5: Test Uninstall/Reinstall**
```bash
brew uninstall claude-whisper
brew install ./Formula/claude-whisper.rb
```

### Phase 5: Prepare Initial Release

**Step 5.1: Create Git Tag in Main Repository**
```bash
cd /Users/sidhu/claude-whisper

# Ensure version is correct in pyproject.toml
# version = "0.1.0"

git add pyproject.toml
git commit -m "Prepare v0.1.0 release"
git push origin main

git tag -a v0.1.0 -m "Release v0.1.0 - Initial Homebrew distribution"
git push origin v0.1.0
```

**Step 5.2: Create GitHub Release**
```bash
gh release create v0.1.0 \
  --title "claude-whisper v0.1.0" \
  --notes "Initial release with Homebrew support

Features:
- Push-to-talk voice interface
- MLX Whisper integration
- Claude Agent SDK integration
- Desktop notifications
- Configurable hotkeys

Installation via Homebrew:
\`\`\`bash
brew tap Ashton-Sidhu/claude-whisper
brew install claude-whisper
\`\`\`
"
```

### Phase 6: Publish Formula

**Step 6.1: Commit Formula**
```bash
cd homebrew-claude-whisper
git add Formula/claude-whisper.rb README.md
git commit -m "Add claude-whisper formula v0.1.0"
git push origin main
```

**Step 6.2: Test from Tap**
```bash
# Uninstall local version
brew uninstall claude-whisper

# Install from tap
brew tap Ashton-Sidhu/claude-whisper
brew install claude-whisper

# Verify installation
claude-whisper --help
which claude-whisper
```

### Phase 7: Update Main Repository Documentation

**Step 7.1: Update README.md**

Add Homebrew installation as primary method for macOS in `/Users/sidhu/claude-whisper/README.md`:

```markdown
## Installation

### macOS (Recommended)

Install via Homebrew:

```bash
brew tap Ashton-Sidhu/claude-whisper
brew install claude-whisper
```

### Development Installation

For development or non-Homebrew installation:

1. Install system dependencies:
```bash
make install-deps
```

2. Install the package:
```bash
uv sync
```
```

**Step 7.2: Add Homebrew Update Instructions**

Add section for users:
```markdown
## Updating

Update via Homebrew:
```bash
brew upgrade claude-whisper
```
```

## Version Management

### Releasing New Versions

**Step 1: Update Version**
```bash
cd /Users/sidhu/claude-whisper
# Edit pyproject.toml: version = "0.2.0"
git commit -am "Bump version to 0.2.0"
git push origin main
```

**Step 2: Create Tag and Release**
```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
gh release create v0.2.0 --generate-notes
```

**Step 3: Calculate New Hash**
```bash
curl -sL https://github.com/Ashton-Sidhu/claude-whisper/archive/refs/tags/v0.2.0.tar.gz | sha256sum
```

**Step 4: Update Formula**
```bash
cd homebrew-claude-whisper
# Edit Formula/claude-whisper.rb:
#   - url: update to v0.2.0
#   - sha256: update with new hash
git commit -am "claude-whisper 0.2.0"
git push origin main
```

**Step 5: Test Update**
```bash
brew upgrade claude-whisper
brew test claude-whisper
```

### Updating Dependencies

When Python dependencies change:

**Step 1: Regenerate Resources**
```bash
cd /Users/sidhu/claude-whisper
poet -f claude-whisper > /tmp/resources.rb
```

**Step 2: Update Formula**
```bash
cd homebrew-claude-whisper
# Replace resource stanzas in Formula/claude-whisper.rb
# with content from /tmp/resources.rb
```

**Step 3: Test**
```bash
brew reinstall --build-from-source claude-whisper
brew test claude-whisper
```

## Optional: Automation

### Automated Release Workflow

Create `.github/workflows/release.yml` in main repository:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Create GitHub Release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh release create ${{ github.ref_name }} \
            --title "${{ github.ref_name }}" \
            --generate-notes

      - name: Calculate tarball hash
        id: hash
        run: |
          HASH=$(curl -sL https://github.com/${{ github.repository }}/archive/refs/tags/${{ github.ref_name }}.tar.gz | sha256sum | cut -d' ' -f1)
          echo "sha256=$HASH" >> $GITHUB_OUTPUT

      - name: Notify about formula update
        run: |
          echo "Update formula with:"
          echo "  url: https://github.com/${{ github.repository }}/archive/refs/tags/${{ github.ref_name }}.tar.gz"
          echo "  sha256: ${{ steps.hash.outputs.sha256 }}"
```

This automates release creation and provides the hash for formula updates.

## User Installation Flow

Once published, users will install with:

```bash
# Tap the repository
brew tap Ashton-Sidhu/claude-whisper

# Install the package
brew install claude-whisper

# Set up API key
export ANTHROPIC_API_KEY="your-key"

# Run the tool
claude-whisper /path/to/project
```

Homebrew will automatically:
1. Install portaudio system dependency
2. Install Python 3.13 if not present
3. Create isolated virtualenv
4. Install all Python dependencies
5. Create `claude-whisper` command in PATH
6. Display caveats about required permissions

## Installation Structure

Files will be installed in:

```
/opt/homebrew/Cellar/claude-whisper/0.1.0/
├── bin/claude-whisper              # Entry point
└── libexec/
    └── lib/python3.13/site-packages/
        ├── claude_whisper/         # Main package
        ├── claude_agent_sdk/       # Dependencies
        ├── mlx_whisper/
        └── ...

/opt/homebrew/bin/claude-whisper    # Symlink to cellar
```

## Critical Files to Modify

1. **`/Users/sidhu/claude-whisper/pyproject.toml`**
   - Update build-system to use hatchling instead of uv_build
   - Maintain current version number

2. **New: `homebrew-claude-whisper/Formula/claude-whisper.rb`**
   - Core formula defining installation process
   - Generated with poet for resource stanzas

3. **`/Users/sidhu/claude-whisper/README.md`**
   - Add Homebrew installation instructions
   - Make it the primary installation method for macOS

4. **New: `homebrew-claude-whisper/README.md`**
   - Tap documentation with quick install guide

## Testing Checklist

- [ ] Formula installs successfully
- [ ] `brew test claude-whisper` passes
- [ ] `brew audit --strict claude-whisper` passes
- [ ] `claude-whisper --help` works
- [ ] Command available in PATH
- [ ] Portaudio dependency installed
- [ ] Python dependencies isolated in virtualenv
- [ ] Uninstall/reinstall works
- [ ] Works on clean macOS system

## Success Criteria

- Users can install with single command: `brew install Ashton-Sidhu/claude-whisper/claude-whisper`
- All dependencies automatically resolved
- Command immediately available after install
- Clear documentation for required permissions
- Easy version updates via `brew upgrade`

## Key Benefits

1. **Automatic dependency management** - portaudio installed automatically
2. **Isolated environment** - No conflicts with system Python
3. **Easy updates** - `brew upgrade` handles everything
4. **Discoverable** - Standard Homebrew workflow
5. **Uninstall support** - Clean removal with `brew uninstall`
6. **Version pinning** - Users can install specific versions
7. **Native macOS integration** - Follows platform conventions
