#!/usr/bin/env bash
set -euo pipefail

# Helper to install fonts for tourism guide PDF generation.
# Includes CJK fonts for Japanese/Chinese/Korean and professional Latin fonts.
# Usage: ./scripts/install-fonts.sh

echo "🔤 Installing fonts for tourism guide PDF generation..."

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "📦 Detected macOS - using Homebrew..."
  
  # Check for Homebrew
  if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install it first: https://brew.sh"
    exit 1
  fi
  
  # Tap the fonts cask if not already tapped
  if ! brew tap | grep -q "homebrew/cask-fonts"; then
    echo "📦 Adding Homebrew fonts tap..."
    if ! brew tap homebrew/cask-fonts; then
      echo "⚠️  Failed to add fonts tap. Some fonts may not be available."
    fi
  fi
  
  echo "📦 Installing CJK fonts (Japanese/Chinese/Korean)..."
  brew list --cask font-noto-sans-cjk 2>/dev/null || brew install --cask font-noto-sans-cjk
  brew list --cask font-noto-serif-cjk 2>/dev/null || brew install --cask font-noto-serif-cjk
  
  echo "📦 Installing professional Latin fonts..."
  brew list --cask font-eb-garamond 2>/dev/null || brew install --cask font-eb-garamond
  brew list --cask font-source-sans-pro 2>/dev/null || brew install --cask font-source-sans-pro
  brew list --cask font-montserrat 2>/dev/null || brew install --cask font-montserrat
  
  echo "✅ All fonts installed successfully!"
  echo "   - Noto Sans CJK (Japanese/Chinese/Korean)"
  echo "   - Noto Serif CJK (Japanese/Chinese/Korean)"
  echo "   - EB Garamond (Professional serif)"
  echo "   - Source Sans Pro (Clean sans-serif)"
  echo "   - Montserrat (Modern sans-serif)"

elif [[ -f /etc/debian_version ]] || command -v apt-get &> /dev/null; then
  echo "📦 Detected Debian/Ubuntu - using apt..."
  
  echo "📦 Installing CJK fonts..."
  sudo apt-get update
  sudo apt-get install -y fonts-noto-cjk fonts-noto-cjk-extra
  
  echo "📦 Installing Latin fonts..."
  sudo apt-get install -y fonts-ebgaramond fonts-dejavu
  
  # Try to install additional fonts if available
  sudo apt-get install -y fonts-liberation fonts-lato 2>/dev/null || true
  
  # Refresh font cache
  fc-cache -fv
  
  echo "✅ Fonts installed successfully!"
  echo "   - Noto CJK (Japanese/Chinese/Korean)"
  echo "   - EB Garamond (Professional serif)"
  echo "   - DejaVu (Fallback fonts)"

elif [[ -f /etc/fedora-release ]] || command -v dnf &> /dev/null; then
  echo "📦 Detected Fedora/RHEL - using dnf..."
  
  echo "📦 Installing CJK fonts..."
  sudo dnf install -y google-noto-sans-cjk-fonts google-noto-serif-cjk-fonts
  
  echo "📦 Installing Latin fonts..."
  sudo dnf install -y dejavu-fonts-all liberation-fonts
  
  # Refresh font cache
  fc-cache -fv
  
  echo "✅ Fonts installed successfully!"

elif [[ -f /etc/arch-release ]] || command -v pacman &> /dev/null; then
  echo "📦 Detected Arch Linux - using pacman..."
  
  echo "📦 Installing CJK fonts..."
  sudo pacman -S --noconfirm noto-fonts-cjk
  
  echo "📦 Installing Latin fonts..."
  sudo pacman -S --noconfirm ttf-dejavu ttf-liberation
  
  # Refresh font cache
  fc-cache -fv
  
  echo "✅ Fonts installed successfully!"

else
  echo "⚠️  Unknown OS. Please install fonts manually:"
  echo ""
  echo "   CJK Fonts (required for Japanese/Chinese/Korean):"
  echo "   - Noto Sans CJK JP: https://www.google.com/get/noto/#sans-jpan"
  echo "   - Noto Serif CJK JP: https://www.google.com/get/noto/#serif-jpan"
  echo ""
  echo "   Latin Fonts (recommended for professional appearance):"
  echo "   - EB Garamond: https://fonts.google.com/specimen/EB+Garamond"
  echo "   - Source Sans Pro: https://fonts.google.com/specimen/Source+Sans+Pro"
  echo "   - Montserrat: https://fonts.google.com/specimen/Montserrat"
  echo ""
  exit 1
fi

echo ""
echo "🎉 Font installation complete!"
echo "   You can now generate PDFs with: uv run guide-cli guide japanese"
