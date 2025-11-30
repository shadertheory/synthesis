<#

.NOTES
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

.SYNOPSIS
Creates a layout and packages a build.

.DESCRIPTION
This script generates a layout from a build and then runs makepkg to create a package.

.PARAMETER Destination
Set the layout path for the build.

.PARAMETER Configuration
Set to 'Debug' or 'Release' to package the build.

.PARAMETER Overwrite
Indicates overwrite of existing content if present.

.PARAMETER Clean
Delete content in the layout directory before copying.

.PARAMETER Quiet
Copy quietly.

.PARAMETER NoPackage
If this is set, then the loose layout is created, but makepkg is not run.

.LINK
https://aka.ms/gdkdocs

#>

param(
    [Parameter(Mandatory)]
    [string]$Destination,
    [string]$Configuration = "Release",
    [switch]$Overwrite,
    [switch]$Clean,
    [switch]$Quiet,
    [switch]$NoPackage
)

$xcopyFlags = "/Y/S"
if($Quiet) {
    $xcopyFlags += "/Q"
}
else {
    $xcopyFlags += "/F"
}

function Copy-Build {

    param(
        [Parameter(Mandatory)]
        [string]$Path,
        [Parameter(Mandatory)]
        [string]$Destination
        )

    $excludefile = Join-Path $PSScriptRoot -Child "PackageLayout.flt"

    $filters | ForEach-Object {
        $files = Join-Path -Path $Path -ChildPath $_
        xcopy $xcopyFlags /EXCLUDE:$excludefile $files $Destination
        if ($LastExitCode -ne 0) {
            Write-Error "Failed copying source files" -ErrorAction Stop
        }
    }
}

$arch = Join-Path -Path $PSScriptRoot -ChildPath "x64"

$config = Join-Path -Path $arch -ChildPath $Configuration

if (-Not (Test-Path $config)) {
    Write-Error ("ERROR: Cannot find a build at " + $config) -ErrorAction Stop
}

$vcpkg = Join-Path -Path $PSScriptRoot -ChildPath "vcpkg_installed"

if(-Not $NoPackage) {
    $makepkg = Get-ChildItem -Path $vcpkg -Recurse -Filter makepkg*.exe

    if ($makepkg.Count -eq 0) {
        Write-Error "ERROR: makepkg not found in VCPKG directory" -ErrorAction Stop
    }
}

$exes = Get-ChildItem -Path $config -Filter *.exe

if ($exes.Count -eq 0) {
    Write-Error "ERROR: No executable found" -ErrorAction Stop
}

$layout = Join-Path $Destination -ChildPath "Loose"

if ((Test-Path $layout) -And (-Not $Overwrite)) {
    Write-Error "ERROR: Destination path exists. Use -Overwrite or change path." -ErrorAction Stop
}

if (-Not $Quiet) {
    Write-Host ("Source: " + $config)
    Write-Host ("Destination: " + $layout)
}

if($Clean) {
    if (-Not $Quiet) {
        Write-Host "Clean..."
    }

    Remove-Item $layout -Recurse -Force -ErrorAction SilentlyContinue | Out-Null
}

New-Item -Path $Destination -Name "Loose" -ItemType Directory -ErrorAction SilentlyContinue | Out-Null

Copy-Build -Path $config -Destination $layout

Set-Location -Path $Destination

if(-Not $NoPackage) {
    <# makepkg genmap /f chunks.xml /d Loose #>
    $makepkgArgs = @( "genmap", "/f", "chunks.xml", "/d", "Loose" )
    Start-Process -FilePath $makepkg[0].FullName -ArgumentList $makepkgArgs -ErrorAction Stop

    <# makepkg pack /f chunks.xml /d Loose /pd . /pc #>
    $makepkgArgs = @( "pack", "/f", "chunks.xml", "/d", "Loose", "/pd", ".", "/pc" )
    Start-Process -FilePath $makepkg[0].FullName -ArgumentList $makepkgArgs -ErrorAction Stop
}
