neuralgym_ps_id=`echo $$`
neuralgym_current_shell=`ps | grep $neuralgym_ps_id | awk '{ print $4 }'`
# echo $neuralgym_current_shell

if [ $neuralgym_current_shell = zsh ]; then
  neuralgym_dir=`dirname $0`
elif [ $neuralgym_current_shell = bash ]; then
  neuralgym_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
else
  echo "Unsupported shell type."
fi

neuralgym_linkdir="`readlink -f $neuralgym_dir/../`"
# echo $neuralgym_linkdir

export PYTHONPATH=`readlink -f $neuralgym_linkdir`:$PYTHONPATH
